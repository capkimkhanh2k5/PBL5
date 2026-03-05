import 'package:flutter/material.dart';

class ScheduleScreen extends StatelessWidget {
  const ScheduleScreen({super.key});

  static const bgColor = Color(0xFFEAF6EE);
  static const primary = Color(0xFF2F6B3D);

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: bgColor,
      appBar: AppBar(
        backgroundColor: bgColor,
        elevation: 0,
        centerTitle: true,
        title: const Text(
          'Lịch Đổ Rác',
          style: TextStyle(
            color: Colors.black87,
            fontWeight: FontWeight.w700,
            fontSize: 22,
          ),
        ),
      ),
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [

            /// NEXT PICKUP CARD
            Container(
              padding: const EdgeInsets.all(20),
              decoration: BoxDecoration(
                borderRadius: BorderRadius.circular(22),
                gradient: const LinearGradient(
                  colors: [
                    Color(0xFFE8F5E9),
                    Color(0xFFD7EDDA),
                  ],
                ),
                boxShadow: [
                  BoxShadow(
                    color: Colors.black.withOpacity(0.08),
                    blurRadius: 18,
                    offset: const Offset(0, 10),
                  ),
                ],
              ),
              child: Column(
                children: [
                  Row(
                    children: [
                      const Icon(Icons.calendar_month,
                          color: primary, size: 28),
                      const SizedBox(width: 12),
                      const Text(
                        "Next Pickup",
                        style: TextStyle(
                            fontSize: 16,
                            fontWeight: FontWeight.w600),
                      ),
                      const Spacer(),

                      /// TEXT THAY VÌ VÒNG TRÒN
                      const Text(
                        "In 2 days",
                        style: TextStyle(
                          color: primary,
                          fontWeight: FontWeight.w600,
                        ),
                      ),
                    ],
                  ),
                  const SizedBox(height: 14),
                  const Text(
                    "Thứ 5 | 7:30 AM",
                    style: TextStyle(
                        fontSize: 20,
                        fontWeight: FontWeight.bold),
                  ),
                  const SizedBox(height: 6),
                  const Text(
                    "Auto-scheduled: Every Monday & Thursday",
                    style: TextStyle(fontSize: 13),
                  ),
                  const SizedBox(height: 18),

                  /// REMIND BUTTON (NHỎ LẠI + CHỮ TRẮNG)
                  Center(
                    child: SizedBox(
                      width: 220,
                      height: 42,
                      child: ElevatedButton.icon(
                        style: ElevatedButton.styleFrom(
                          backgroundColor: primary,
                          foregroundColor: Colors.white,
                          shape: RoundedRectangleBorder(
                            borderRadius: BorderRadius.circular(25),
                          ),
                          elevation: 0,
                        ),
                        onPressed: () {},
                        icon: const Icon(
                          Icons.notifications,
                          size: 18,
                          color: Colors.white,
                        ),
                        label: const Text(
                          "Remind Me",
                          style: TextStyle(
                            fontWeight: FontWeight.w600,
                            color: Colors.white,
                          ),
                        ),
                      ),
                    ),
                  )
                ],
              ),
            ),

            const SizedBox(height: 26),

            const Text(
              "Upcoming Pickups",
              style: TextStyle(
                  fontSize: 18,
                  fontWeight: FontWeight.w700),
            ),

            const SizedBox(height: 14),

            _pickupTile("Mon, Mar 4",
                "7:30 AM · Organic Waste",
                "Tomorrow"),

            _pickupTile("Thu, Mar 7",
                "7:30 AM · Paper & Plastic",
                "In 4 days"),

            _pickupTile("Mon, Mar 11",
                "7:30 AM · Organic Waste",
                "In 7 days"),

            _pickupTile("Thu, Mar 14",
                "7:30 AM · Paper & Plastic",
                "In 11 days"),
          ],
        ),
      ),
    );
  }

  static Widget _pickupTile(
      String date, String subtitle, String badge) {
    return Container(
      margin: const EdgeInsets.only(bottom: 14),
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(18),
        boxShadow: [
          BoxShadow(
            color: Colors.black.withOpacity(0.06),
            blurRadius: 14,
            offset: const Offset(0, 8),
          )
        ],
      ),
      child: Row(
        children: [
          const Icon(Icons.access_time,
              color: primary, size: 28),
          const SizedBox(width: 14),
          Expanded(
            child: Column(
              crossAxisAlignment:
              CrossAxisAlignment.start,
              children: [
                Text(
                  date,
                  style: const TextStyle(
                      fontWeight: FontWeight.w600,
                      fontSize: 15),
                ),
                const SizedBox(height: 4),
                Text(
                  subtitle,
                  style: const TextStyle(fontSize: 13),
                ),
              ],
            ),
          ),
          Container(
            padding: const EdgeInsets.symmetric(
                horizontal: 14, vertical: 6),
            decoration: BoxDecoration(
              color: const Color(0xFFE3F2E6),
              borderRadius: BorderRadius.circular(20),
            ),
            child: Text(
              badge,
              style: const TextStyle(
                  fontSize: 12,
                  fontWeight: FontWeight.w600),
            ),
          )
        ],
      ),
    );
  }
}