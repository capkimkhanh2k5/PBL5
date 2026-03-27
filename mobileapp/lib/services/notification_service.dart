import 'dart:async';
import 'dart:io';

import 'package:flutter/foundation.dart';
import 'package:flutter_local_notifications/flutter_local_notifications.dart';
import 'package:firebase_messaging/firebase_messaging.dart';
import 'package:flutter_timezone/flutter_timezone.dart';
import 'package:timezone/data/latest.dart' as tz;
import 'package:timezone/timezone.dart' as tz;

class NotificationService {
  NotificationService._();

  static final NotificationService instance = NotificationService._();
  static const int pickupReminderNotificationId = 1001;
  static const int alertNotificationId = 2001;
  static const String _alertChannelId = 'system_alert_channel';
  static const String _alertTopic = 'system-alerts';

  final FlutterLocalNotificationsPlugin _plugin =
      FlutterLocalNotificationsPlugin();
  bool _initialized = false;
  bool _fcmInitialized = false;
  bool _alertTopicSubscribed = false;
  bool _topicRetryInProgress = false;

  Future<void> init() async {
    if (_initialized) return;

    tz.initializeTimeZones();
    try {
      final timezoneName = await FlutterTimezone.getLocalTimezone();
      tz.setLocalLocation(tz.getLocation(timezoneName));
    } catch (_) {
      // Fallback to default timezone from package when platform lookup fails.
    }

    const android = AndroidInitializationSettings('@mipmap/ic_launcher');
    const ios = DarwinInitializationSettings();
    const settings = InitializationSettings(android: android, iOS: ios);

    await _plugin.initialize(settings);

    await _plugin
        .resolvePlatformSpecificImplementation<
          AndroidFlutterLocalNotificationsPlugin
        >()
        ?.requestNotificationsPermission();
    await _plugin
        .resolvePlatformSpecificImplementation<
          IOSFlutterLocalNotificationsPlugin
        >()
        ?.requestPermissions(alert: true, badge: true, sound: true);

    _initialized = true;
  }

  Future<void> initFcmAlerts() async {
    if (_fcmInitialized) return;

    await init();

    final messaging = FirebaseMessaging.instance;
    await messaging.requestPermission(alert: true, badge: true, sound: true);
    await messaging.setForegroundNotificationPresentationOptions(
      alert: true,
      badge: true,
      sound: true,
    );

    final subscribed = await _subscribeToAlertTopic(messaging);
    if (!subscribed && Platform.isIOS) {
      unawaited(_retrySubscribeToAlertTopic(messaging));
    }

    FirebaseMessaging.onMessage.listen((RemoteMessage message) async {
      final title =
          message.notification?.title ?? message.data['title'] ?? 'Canh bao he thong';
      final body =
          message.notification?.body ?? message.data['message'] ?? 'Co canh bao moi.';
      await showImmediateAlert(title: title.toString(), body: body.toString());
    });

    _fcmInitialized = true;
  }

  Future<bool> _subscribeToAlertTopic(FirebaseMessaging messaging) async {
    if (_alertTopicSubscribed) return true;

    try {
      if (Platform.isIOS) {
        final apnsToken = await _waitForApnsToken(messaging);
        if (apnsToken == null) {
          debugPrint(
            'FCM: APNS token not ready yet, skip subscribe and retry later.',
          );
          return false;
        }
      }

      await messaging.subscribeToTopic(_alertTopic);
      _alertTopicSubscribed = true;
      return true;
    } catch (e) {
      debugPrint('FCM: subscribe to topic failed: $e');
      return false;
    }
  }

  Future<void> _retrySubscribeToAlertTopic(FirebaseMessaging messaging) async {
    if (_topicRetryInProgress || _alertTopicSubscribed) return;
    _topicRetryInProgress = true;

    try {
      for (var attempt = 0; attempt < 6 && !_alertTopicSubscribed; attempt++) {
        await Future.delayed(const Duration(seconds: 5));
        final subscribed = await _subscribeToAlertTopic(messaging);
        if (subscribed) break;
      }
    } finally {
      _topicRetryInProgress = false;
    }
  }

  Future<String?> _waitForApnsToken(FirebaseMessaging messaging) async {
    for (var attempt = 0; attempt < 10; attempt++) {
      final token = await messaging.getAPNSToken();
      if (token != null && token.isNotEmpty) {
        return token;
      }
      await Future.delayed(const Duration(milliseconds: 500));
    }
    return null;
  }

  Future<void> showImmediateAlert({required String title, required String body}) async {
    await init();

    const details = NotificationDetails(
      android: AndroidNotificationDetails(
        _alertChannelId,
        'System Alerts',
        channelDescription: 'Immediate system alerts from backend',
        importance: Importance.max,
        priority: Priority.high,
      ),
      iOS: DarwinNotificationDetails(),
    );

    await _plugin.show(alertNotificationId, title, body, details);
  }

  Future<void> schedulePickupReminder({
    required DateTime scheduledAt,
    required String title,
    required String body,
  }) async {
    await init();

    final at = tz.TZDateTime.from(scheduledAt, tz.local);
    const details = NotificationDetails(
      android: AndroidNotificationDetails(
        'pickup_reminder_channel',
        'Pickup Reminders',
        channelDescription: 'Reminder notifications for trash pickup schedule',
        importance: Importance.max,
        priority: Priority.high,
      ),
      iOS: DarwinNotificationDetails(),
    );

    await _plugin.zonedSchedule(
      pickupReminderNotificationId,
      title,
      body,
      at,
      details,
      androidScheduleMode: AndroidScheduleMode.exactAllowWhileIdle,
      uiLocalNotificationDateInterpretation:
          UILocalNotificationDateInterpretation.absoluteTime,
      matchDateTimeComponents: null,
    );
  }

  Future<void> cancelPickupReminder() async {
    await init();
    await _plugin.cancel(pickupReminderNotificationId);
  }
}
