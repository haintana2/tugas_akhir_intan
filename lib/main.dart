import 'dart:async';
import 'dart:convert';
import 'dart:io';

import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:http/http.dart' as http;
import 'package:http_parser/http_parser.dart'; // For MediaType
import 'package:image/image.dart' as img;
import 'package:permission_handler/permission_handler.dart';

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();
  final cameras = await availableCameras();
  runApp(MyApp(cameras: cameras));
}

class MyApp extends StatelessWidget {
  final List<CameraDescription> cameras;
  const MyApp({Key? key, required this.cameras}) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      debugShowCheckedModeBanner: false,
      title: 'Realtime Leaf Prediction',
      home: PredictionPage(cameras: cameras),
    );
  }
}

class PredictionPage extends StatefulWidget {
  final List<CameraDescription> cameras;
  const PredictionPage({Key? key, required this.cameras}) : super(key: key);

  @override
  _PredictionPageState createState() => _PredictionPageState();
}

class _PredictionPageState extends State<PredictionPage>
    with WidgetsBindingObserver {
  CameraController? _controller;
  Timer? _timer;
  bool _isProcessing = false;
  String? _prediction = "Waiting for prediction...";
  String? _serverIP;

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addObserver(this);
    _askForServerIP();
  }

  @override
  void dispose() {
    _timer?.cancel();
    _controller?.dispose();
    WidgetsBinding.instance.removeObserver(this);
    super.dispose();
  }

  // Show a dialog to enter the IP address before starting the camera
  Future<void> _askForServerIP() async {
    String? ip = await showDialog<String>(
      context: context,
      barrierDismissible: false, // Force user to enter an IP
      builder: (context) {
        TextEditingController ipController =
            TextEditingController(text: '192.168.100.7');
        return AlertDialog(
          title: const Text("Enter Server IP"),
          content: TextField(
            controller: ipController,
            keyboardType: TextInputType.number,
            decoration: const InputDecoration(
              hintText: "e.g., 192.168.100.7",
            ),
          ),
          actions: [
            TextButton(
              onPressed: () {
                Navigator.of(context).pop(null); // User cancels
              },
              child: const Text("Cancel"),
            ),
            TextButton(
              onPressed: () {
                Navigator.of(context).pop(ipController.text.trim());
              },
              child: const Text("OK"),
            ),
          ],
        );
      },
    );

    if (ip == null || ip.isEmpty) {
      SystemNavigator.pop(); // Exit app if no IP is entered
    } else {
      setState(() {
        _serverIP = ip;
      });
      _initializeCamera();
    }
  }

  @override
  void didChangeAppLifecycleState(AppLifecycleState state) {
    if (state == AppLifecycleState.inactive ||
        state == AppLifecycleState.paused ||
        state == AppLifecycleState.detached) {
      _timer?.cancel();
      _controller?.dispose();
      _controller = null;
    } else if (state == AppLifecycleState.resumed) {
      if (_controller == null || !_controller!.value.isInitialized) {
        _initializeCamera();
      }
    }
  }

  Future<void> _initializeCamera() async {
    final permissionStatus = await Permission.camera.request();
    if (!permissionStatus.isGranted) {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(content: Text('Camera permission is required')),
        );
      }
      return;
    }

    if (widget.cameras.isEmpty) {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(content: Text('No camera found')),
        );
      }
      return;
    }

    _controller = CameraController(
      widget.cameras[0],
      ResolutionPreset.medium,
      imageFormatGroup: ImageFormatGroup.jpeg,
    );
    await _controller!.initialize();
    await _controller!.setFlashMode(FlashMode.off); // Disable flash

    if (!mounted) return;
    setState(() {});

    _timer = Timer.periodic(const Duration(seconds: 3), (timer) {
      _captureAndPredict();
    });
  }

  Future<void> _captureAndPredict() async {
    if (_controller == null ||
        !_controller!.value.isInitialized ||
        _isProcessing ||
        _serverIP == null) {
      return;
    }

    try {
      setState(() {
        _isProcessing = true;
      });

      final XFile file = await _controller!.takePicture();
      File imageFile = File(file.path);
      List<int> imageBytes = await imageFile.readAsBytes();

      img.Image? capturedImage = img.decodeImage(imageBytes);
      if (capturedImage == null)
        throw Exception("Could not decode captured image");

      int width = capturedImage.width;
      int height = capturedImage.height;
      int squareSize = width < height ? width : height;
      int offsetX = ((width - squareSize) / 2).round();
      int offsetY = ((height - squareSize) / 2).round();
      img.Image croppedImage =
          img.copyCrop(capturedImage, offsetX, offsetY, squareSize, squareSize);
      img.Image resizedImage =
          img.copyResize(croppedImage, width: 320, height: 320);
      List<int> jpeg = img.encodeJpg(resizedImage);

      var uri = Uri.parse('http://$_serverIP:5000/predict');
      var request = http.MultipartRequest('POST', uri);
      request.files.add(http.MultipartFile.fromBytes(
        'image',
        jpeg,
        filename: 'captured.jpg',
        contentType: MediaType('image', 'jpeg'),
      ));

      var response = await request.send();
      if (response.statusCode == 200) {
        String responseBody = await response.stream.bytesToString();
        var result = jsonDecode(responseBody);
        setState(() {
          if (result.containsKey("class") && result.containsKey("confidence")) {
            _prediction =
                "Class: ${result['class']}\nConfidence: ${result['confidence']}";
          } else if (result.containsKey("error")) {
            _prediction = "Error: ${result['error']}";
          } else {
            _prediction = "Unexpected response";
          }
        });
      } else {
        setState(() {
          _prediction = "Error: Server responded with ${response.statusCode}";
        });
      }
    } catch (e) {
      setState(() {
        _prediction = "Error: $e";
      });
    } finally {
      setState(() {
        _isProcessing = false;
      });
    }
  }

  Future<bool> _onWillPop() async {
    _timer?.cancel();
    await _controller?.dispose();
    SystemNavigator.pop();
    return true;
  }

  @override
  Widget build(BuildContext context) {
    return WillPopScope(
      onWillPop: _onWillPop,
      child: Scaffold(
        appBar: AppBar(
          title: const Text('Realtime Leaf Prediction'),
          centerTitle: true,
        ),
        body: _controller == null || !_controller!.value.isInitialized
            ? const Center(child: CircularProgressIndicator())
            : Column(
                children: [
                  AspectRatio(
                    aspectRatio: 1,
                    child: CameraPreview(_controller!),
                  ),
                  const SizedBox(height: 16),
                  Text(
                    _prediction ?? "",
                    textAlign: TextAlign.center,
                    style: const TextStyle(
                        fontSize: 18, fontWeight: FontWeight.bold),
                  ),
                ],
              ),
      ),
    );
  }
}
