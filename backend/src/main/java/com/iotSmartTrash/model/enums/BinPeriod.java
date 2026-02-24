package com.iotSmartTrash.model.enums;

public enum BinPeriod {
    H00,
    H06,
    H12,
    H18;

    /** Trả về label dạng string để lưu vào Firestore ("00h", "06h", ...) */
    public String toLabel() {
        return this.name().substring(1) + "h";
    }

    /** Parse từ label string trong Firestore ("00h", "06h", ...) sang enum */
    public static BinPeriod fromLabel(String label) {
        if (label == null)
            return null;
        return switch (label) {
            case "00h" -> H00;
            case "06h" -> H06;
            case "12h" -> H12;
            case "18h" -> H18;
            default -> throw new IllegalArgumentException("Unknown period: " + label);
        };
    }

    /** Map from clock hour to period */
    public static BinPeriod fromHour(int hour) {
        if (hour < 6)
            return H00;
        if (hour < 12)
            return H06;
        if (hour < 18)
            return H12;
        return H18;
    }
}
