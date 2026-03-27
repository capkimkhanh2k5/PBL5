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
    await messaging.subscribeToTopic(_alertTopic);

    FirebaseMessaging.onMessage.listen((RemoteMessage message) async {
      final title =
          message.notification?.title ?? message.data['title'] ?? 'Canh bao he thong';
      final body =
          message.notification?.body ?? message.data['message'] ?? 'Co canh bao moi.';
      await showImmediateAlert(title: title.toString(), body: body.toString());
    });

    _fcmInitialized = true;
  }

  Future<void> showImmediateAlert({required String title, required String body}) async {
    await init();

    final details = NotificationDetails(
      android: const AndroidNotificationDetails(
        _alertChannelId,
        'System Alerts',
        channelDescription: 'Immediate system alerts from backend',
        importance: Importance.max,
        priority: Priority.high,
      ),
      iOS: const DarwinNotificationDetails(),
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
    final details = NotificationDetails(
      android: AndroidNotificationDetails(
        'pickup_reminder_channel',
        'Pickup Reminders',
        channelDescription: 'Reminder notifications for trash pickup schedule',
        importance: Importance.max,
        priority: Priority.high,
      ),
      iOS: const DarwinNotificationDetails(),
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
