import 'package:firebase_core/firebase_core.dart' show FirebaseOptions;
import 'package:flutter/foundation.dart'
    show defaultTargetPlatform, kIsWeb, TargetPlatform;

class DefaultFirebaseOptions {
  static FirebaseOptions get currentPlatform {
    if (kIsWeb) {
      throw UnsupportedError('Web platform is not configured.');
    }
    switch (defaultTargetPlatform) {
      case TargetPlatform.android:
        return android;
      case TargetPlatform.iOS:
        return ios;
      case TargetPlatform.macOS:
        throw UnsupportedError('macOS platform is not configured.');
      case TargetPlatform.windows:
        throw UnsupportedError('Windows platform is not configured.');
      case TargetPlatform.linux:
        throw UnsupportedError('Linux platform is not configured.');
      default:
        throw UnsupportedError('Unsupported platform.');
    }
  }

  static const FirebaseOptions android = FirebaseOptions(
    apiKey: 'AIzaSyBUoRROviLSkM3XVpeyOW9pZFgJ955FRn8',
    appId: '1:86746937846:android:59a2948882c713bf67efd7',
    messagingSenderId: '86746937846',
    projectId: 'pbl5-f21e6',
    databaseURL: 'https://pbl5-f21e6-default-rtdb.asia-southeast1.firebasedatabase.app',
    storageBucket: 'pbl5-f21e6.firebasestorage.app',
  );

  static const FirebaseOptions ios = FirebaseOptions(
    apiKey: 'AIzaSyBJ1sdlKoH27EhU-52o66GBMke15cEBNOE',
    appId: '1:86746937846:ios:28a3ab2ab7cb4c8c67efd7',
    messagingSenderId: '86746937846',
    projectId: 'pbl5-f21e6',
    databaseURL: 'https://pbl5-f21e6-default-rtdb.asia-southeast1.firebasedatabase.app',
    storageBucket: 'pbl5-f21e6.firebasestorage.app',
    iosBundleId: 'com.example.pbl5Flutter',
    iosClientId: '86746937846-k3e9i2aosaj2rcp3f6pf99qcl9rkblgi.apps.googleusercontent.com',
  );
}
