import 'dart:convert';
import 'package:http/http.dart' as http;

class GroqService {
  static const String apiKey = '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';

  Future<String> send(String message) async {
    final res = await http.post(
      Uri.parse('https://api.groq.com/openai/v1/chat/completions'),
      headers: {
        'Content-Type': 'application/json',
        'Authorization': 'Bearer $apiKey',
      },
      body: jsonEncode({
        "model": "llama-3.1-8b-instant",
        "messages": [
          {
            "role": "system",
            "content": """
You are an AI assistant for a Smart Trash App.

You can answer questions about:
- bins
- trash bins
- smart bins
- bin fill levels
- nearly full bins
- pickup schedule
- collection schedule
- disposal history
- collection routes
- route optimization

These questions are related to the Smart Trash App:
- Show bins that are nearly full
- View today's collection schedule
- Track disposal history
- Optimize collection route

Always use the user's provided data and instructions first.

If the user asks about something outside the Smart Trash App, say:
I can only help with the trash app.

"""
          },
          {
            "role": "user",
            "content": message
          }
        ]
      }),
    );

    print("statusCode: ${res.statusCode}");
    print("body: ${res.body}");

    final data = jsonDecode(res.body);

    if (res.statusCode != 200) {
      final errorMessage = data["error"]?["message"] ?? "Unknown API error";
      throw Exception("Groq API error: $errorMessage");
    }

    final choices = data["choices"];
    if (choices == null || choices.isEmpty) {
      throw Exception("No choices returned from Groq");
    }

    final content = choices[0]["message"]?["content"];
    if (content == null) {
      throw Exception("No message content returned from Groq");
    }

    return content.toString();
  }
}