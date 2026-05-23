import 'package:flutter/material.dart';
import '../services/api_service.dart';

class RecycleStatisticsScreen extends StatefulWidget {
  const RecycleStatisticsScreen({
    super.key,
    required this.apiService,
  });

  final ApiService apiService;

  @override
  State<RecycleStatisticsScreen> createState() =>
      _RecycleStatisticsScreenState();
}

class _RecycleStatisticsScreenState extends State<RecycleStatisticsScreen> {
  int _periodIndex = 0; // 0 = This week, 1 = Last 7 days, 2 = Custom
  int _viewIndex = 0; // 0 = All bins, 1 = By bin

  String? _selectedBinId;

  bool _isLoading = true;
  String? _error;

  List<_BinOption> _bins = [];
  _RecycleStats? _currentStats;
  _RecycleStats? _allStatsForShare;

  DateTime? _customStart;
  DateTime? _customEnd;

  static const darkGreen = Color(0xFF0B5D1E);
  static const bgColor = Color(0xFFFBFDF7);
  static const softGreen = Color(0xFFEAF6C8);

  @override
  void initState() {
    super.initState();
    _loadInitialData();
  }

  Future<void> _loadInitialData() async {
    setState(() {
      _isLoading = true;
      _error = null;
    });

    try {
      final binsRaw = await widget.apiService.getAllBins();

      final bins = binsRaw.map<_BinOption>((item) {
        final id = (item['id'] ?? '').toString();
        final name = (item['name'] ?? id).toString();

        final location = (item['locationDescription'] ??
            item['location_description'] ??
            item['location'] ??
            '')
            .toString();

        return _BinOption(
          id: id,
          name: name,
          location: location,
        );
      }).where((bin) => bin.id.isNotEmpty).toList();

      if (!mounted) return;

      setState(() {
        _bins = bins;

        if (_selectedBinId == null && bins.isNotEmpty) {
          _selectedBinId = bins.first.id;
        }
      });

      await _loadStatistics();
    } catch (e) {
      if (!mounted) return;

      setState(() {
        _error = 'Failed to load bins.';
        _isLoading = false;
      });
    }
  }

  Future<void> _showBinPicker() async {
    if (_bins.isEmpty) return;

    final selectedId = await showModalBottomSheet<String>(
      context: context,
      backgroundColor: Colors.transparent,
      isScrollControlled: true,
      builder: (context) {
        return Container(
          constraints: BoxConstraints(
            maxHeight: MediaQuery.of(context).size.height * 0.62,
          ),
          decoration: const BoxDecoration(
            color: Colors.white,
            borderRadius: BorderRadius.vertical(
              top: Radius.circular(28),
            ),
          ),
          child: SafeArea(
            top: false,
            child: Padding(
              padding: const EdgeInsets.fromLTRB(18, 12, 18, 20),
              child: Column(
                mainAxisSize: MainAxisSize.min,
                children: [
                  Container(
                    width: 46,
                    height: 5,
                    decoration: BoxDecoration(
                      color: Colors.black.withOpacity(0.16),
                      borderRadius: BorderRadius.circular(20),
                    ),
                  ),

                  const SizedBox(height: 18),

                  const Align(
                    alignment: Alignment.centerLeft,
                    child: Text(
                      'Select bin',
                      style: TextStyle(
                        fontSize: 22,
                        fontWeight: FontWeight.w900,
                      ),
                    ),
                  ),

                  const SizedBox(height: 16),

                  Flexible(
                    child: ListView.separated(
                      shrinkWrap: true,
                      itemCount: _bins.length,
                      separatorBuilder: (_, __) => const SizedBox(height: 10),
                      itemBuilder: (context, index) {
                        final bin = _bins[index];
                        final selected = bin.id == _selectedBinId;

                        return InkWell(
                          borderRadius: BorderRadius.circular(18),
                          onTap: () => Navigator.pop(context, bin.id),
                          child: Container(
                            padding: const EdgeInsets.all(14),
                            decoration: BoxDecoration(
                              color: selected
                                  ? const Color(0xFFEAF6C8)
                                  : Colors.white,
                              borderRadius: BorderRadius.circular(18),
                              border: Border.all(
                                color: selected
                                    ? darkGreen
                                    : Colors.black.withOpacity(0.08),
                                width: selected ? 1.4 : 1,
                              ),
                              boxShadow: [
                                BoxShadow(
                                  color: Colors.black.withOpacity(0.035),
                                  blurRadius: 12,
                                  offset: const Offset(0, 5),
                                ),
                              ],
                            ),
                            child: Row(
                              children: [
                                Container(
                                  width: 42,
                                  height: 42,
                                  decoration: BoxDecoration(
                                    color: selected
                                        ? darkGreen
                                        : const Color(0xFFF4F4F4),
                                    shape: BoxShape.circle,
                                  ),
                                  child: Icon(
                                    Icons.delete_outline,
                                    color: selected
                                        ? Colors.white
                                        : Colors.black87,
                                    size: 22,
                                  ),
                                ),

                                const SizedBox(width: 12),

                                Expanded(
                                  child: Column(
                                    crossAxisAlignment: CrossAxisAlignment.start,
                                    children: [
                                      Text(
                                        bin.id,
                                        style: const TextStyle(
                                          fontSize: 16,
                                          fontWeight: FontWeight.w900,
                                          color: Colors.black,
                                        ),
                                      ),
                                      const SizedBox(height: 4),
                                      Text(
                                        bin.location.isEmpty
                                            ? 'No location'
                                            : bin.location,
                                        maxLines: 1,
                                        overflow: TextOverflow.ellipsis,
                                        style: TextStyle(
                                          fontSize: 13,
                                          color: Colors.black.withOpacity(0.55),
                                          fontWeight: FontWeight.w600,
                                        ),
                                      ),
                                    ],
                                  ),
                                ),

                                if (selected)
                                  const Icon(
                                    Icons.check_circle,
                                    color: darkGreen,
                                    size: 24,
                                  ),
                              ],
                            ),
                          ),
                        );
                      },
                    ),
                  ),
                ],
              ),
            ),
          ),
        );
      },
    );

    if (selectedId == null || selectedId == _selectedBinId) return;

    setState(() {
      _selectedBinId = selectedId;
    });

    await _loadStatistics();
  }

  Future<void> _loadStatistics() async {
    setState(() {
      _isLoading = true;
      _error = null;
    });

    try {
      final range = _getDateRange();
      final isByBin = _viewIndex == 1;

      final startDate = _formatDate(range.start);
      final endDate = _formatDate(range.end);

      final futures = <Future<Map<String, dynamic>>>[
        widget.apiService.getWeeklyRecycleStatistics(
          binId: isByBin ? _selectedBinId : null,
          startDate: startDate,
          endDate: endDate,
        ),
      ];

      if (isByBin) {
        futures.add(
          widget.apiService.getWeeklyRecycleStatistics(
            startDate: startDate,
            endDate: endDate,
          ),
        );
      }

      final results = await Future.wait(futures);

      final currentRaw = results[0];
      final allRaw = isByBin ? results[1] : null;

      if (!mounted) return;

      setState(() {
        _currentStats = _RecycleStats.fromJson(currentRaw);
        _allStatsForShare = allRaw == null
            ? null
            : _RecycleStats.fromJson(allRaw);
        _isLoading = false;
      });
    } catch (e) {
      if (!mounted) return;

      setState(() {
        _error = 'Failed to load statistics from backend.';
        _isLoading = false;
      });
    }
  }

  _DateRange _getDateRange() {
    final now = DateTime.now();
    final today = DateTime(now.year, now.month, now.day);

    // Monday của tuần hiện tại
    final startOfThisWeek = today.subtract(
      Duration(days: today.weekday - 1),
    );

    // This week: từ thứ Hai tuần này đến hôm nay
    if (_periodIndex == 0) {
      return _DateRange(
        start: startOfThisWeek,
        end: today,
      );
    }

    // Last week: từ thứ Hai tuần trước đến Chủ nhật tuần trước
    if (_periodIndex == 1) {
      final startOfLastWeek = startOfThisWeek.subtract(
        const Duration(days: 7),
      );

      final endOfLastWeek = startOfThisWeek.subtract(
        const Duration(days: 1),
      );

      return _DateRange(
        start: startOfLastWeek,
        end: endOfLastWeek,
      );
    }

    // Custom range
    if (_periodIndex == 2 && _customStart != null && _customEnd != null) {
      return _DateRange(
        start: _customStart!,
        end: _customEnd!,
      );
    }

    return _DateRange(
      start: startOfThisWeek,
      end: today,
    );
  }

  Future<void> _pickCustomRange() async {
    final now = DateTime.now();

    final picked = await showDateRangePicker(
      context: context,
      firstDate: DateTime(now.year - 2),
      lastDate: DateTime(now.year + 1),
      initialDateRange: DateTimeRange(
        start: now.subtract(const Duration(days: 6)),
        end: now,
      ),
    );

    if (picked == null) return;

    setState(() {
      _periodIndex = 2;
      _customStart = DateTime(
        picked.start.year,
        picked.start.month,
        picked.start.day,
      );
      _customEnd = DateTime(
        picked.end.year,
        picked.end.month,
        picked.end.day,
      );
    });

    await _loadStatistics();
  }

  String _formatDate(DateTime date) {
    final y = date.year.toString().padLeft(4, '0');
    final m = date.month.toString().padLeft(2, '0');
    final d = date.day.toString().padLeft(2, '0');

    return '$y-$m-$d';
  }

  String _selectedBinDisplayText() {
    if (_selectedBinId == null) return 'Select a bin';

    final selected = _bins.where((bin) => bin.id == _selectedBinId).toList();

    if (selected.isEmpty) return _selectedBinId!;

    final bin = selected.first;

    if (bin.location.isEmpty) {
      return bin.id;
    }

    return '${bin.id} - ${bin.location}';
  }

  String _selectedBinLabel() {
    if (_selectedBinId == null) return 'Unknown bin';

    final selected = _bins.where((bin) => bin.id == _selectedBinId).toList();

    if (selected.isEmpty) return _selectedBinId!;

    return selected.first.id;
  }

  _RecycleDay? _getBestDay(List<_RecycleDay> data) {
    if (data.isEmpty) return null;

    final nonZero = data.where((e) => e.liters > 0).toList();

    if (nonZero.isEmpty) return null;

    return nonZero.reduce(
          (a, b) => a.liters >= b.liters ? a : b,
    );
  }

  double _calculateShare({
    required bool isByBin,
    required double total,
    required double allTotal,
  }) {
    if (!isByBin || allTotal <= 0) return 0.0;

    return (total / allTotal) * 100.0;
  }

  @override
  Widget build(BuildContext context) {
    final isByBin = _viewIndex == 1;
    final stats = _currentStats;

    final data = stats?.days ?? [];
    final total = stats?.totalLiters ?? 0.0;
    final percentChange = stats?.percentChange ?? 0;

    final maxValue = data.isNotEmpty
        ? data.map((e) => e.liters).reduce((a, b) => a > b ? a : b)
        : 1.0;

    final safeMaxValue = maxValue <= 0 ? 1.0 : maxValue;

    final bestDay = _getBestDay(data);
    final activeDays = data.where((e) => e.liters > 0).length;
    final average = activeDays == 0 ? 0.0 : total / activeDays;

    final share = _calculateShare(
      isByBin: isByBin,
      total: total,
      allTotal: _allStatsForShare?.totalLiters ?? 0.0,
    );

    return Scaffold(
      backgroundColor: bgColor,
      body: SafeArea(
        child: _isLoading
            ? const Center(
          child: CircularProgressIndicator(
            color: darkGreen,
          ),
        )
            : _error != null
            ? Center(
          child: Padding(
            padding: const EdgeInsets.all(24),
            child: Column(
              mainAxisSize: MainAxisSize.min,
              children: [
                Text(
                  _error!,
                  textAlign: TextAlign.center,
                  style: const TextStyle(
                    color: Colors.red,
                    fontWeight: FontWeight.w700,
                  ),
                ),
                const SizedBox(height: 12),
                ElevatedButton(
                  onPressed: _loadStatistics,
                  child: const Text('Retry'),
                ),
              ],
            ),
          ),
        )
            : SingleChildScrollView(
          padding: const EdgeInsets.fromLTRB(18, 14, 18, 28),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Row(
                children: [
                  IconButton(
                    onPressed: () => Navigator.pop(context),
                    icon: const Icon(Icons.arrow_back, size: 28),
                  ),
                  const SizedBox(width: 8),
                  const Expanded(
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Text(
                          'Recycle Statistics',
                          style: TextStyle(
                            fontSize: 26,
                            fontWeight: FontWeight.w900,
                          ),
                        ),
                        SizedBox(height: 2),
                        Text(
                          'Overview of recycled waste',
                          style: TextStyle(
                            color: Colors.black54,
                            fontWeight: FontWeight.w600,
                          ),
                        ),
                      ],
                    ),
                  ),
                  const Icon(Icons.info_outline, color: darkGreen),
                ],
              ),

              const SizedBox(height: 22),

              Row(
                children: [
                  _PeriodChip(
                    label: 'This week',
                    selected: _periodIndex == 0,
                    onTap: () {
                      setState(() {
                        _periodIndex = 0;
                        _customStart = null;
                        _customEnd = null;
                      });

                      _loadStatistics();
                    },
                  ),
                  const SizedBox(width: 10),
                  _PeriodChip(
                    label: 'Last week',
                    selected: _periodIndex == 1,
                    onTap: () {
                      setState(() {
                        _periodIndex = 1;
                        _customStart = null;
                        _customEnd = null;
                      });

                      _loadStatistics();
                    },
                  ),
                  const SizedBox(width: 10),
                  _PeriodChip(
                    label: 'Custom',
                    selected: _periodIndex == 2,
                    onTap: _pickCustomRange,
                  ),
                ],
              ),

              const SizedBox(height: 18),

              _TotalCard(
                title: isByBin
                    ? 'Total recycled (${_selectedBinLabel()})'
                    : 'Total recycled (All bins)',
                total: total,
                subtitle:
                '${percentChange >= 0 ? '+' : ''}$percentChange% compared with previous period',
              ),

              const SizedBox(height: 18),

              Row(
                children: [
                  const Text(
                    'View by',
                    style: TextStyle(
                      fontSize: 16,
                      fontWeight: FontWeight.w900,
                    ),
                  ),
                  const SizedBox(width: 12),
                  Expanded(
                    child: _SegmentButton(
                      leftText: 'All bins',
                      rightText: 'By bin',
                      selectedIndex: _viewIndex,
                      onChanged: (index) {
                        setState(() => _viewIndex = index);

                        _loadStatistics();
                      },
                    ),
                  ),
                ],
              ),

              if (isByBin) ...[
                const SizedBox(height: 16),
                const Text(
                  'Select bin',
                  style: TextStyle(
                    fontSize: 15,
                    fontWeight: FontWeight.w900,
                  ),
                ),
                const SizedBox(height: 8),

                GestureDetector(
                  onTap: _showBinPicker,
                  child: Container(
                    width: double.infinity,
                    padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 15),
                    decoration: BoxDecoration(
                      color: Colors.white,
                      borderRadius: BorderRadius.circular(18),
                      border: Border.all(
                        color: darkGreen.withOpacity(0.55),
                        width: 1.1,
                      ),
                      boxShadow: [
                        BoxShadow(
                          color: Colors.black.withOpacity(0.04),
                          blurRadius: 12,
                          offset: const Offset(0, 5),
                        ),
                      ],
                    ),
                    child: Row(
                      children: [
                        Expanded(
                          child: Text(
                            _selectedBinDisplayText(),
                            maxLines: 1,
                            overflow: TextOverflow.ellipsis,
                            style: const TextStyle(
                              fontSize: 15.5,
                              fontWeight: FontWeight.w800,
                              color: Colors.black87,
                            ),
                          ),
                        ),
                        const SizedBox(width: 8),
                        const Icon(
                          Icons.keyboard_arrow_down_rounded,
                          size: 26,
                          color: Colors.black54,
                        ),
                      ],
                    ),
                  ),
                ),
              ],

              const SizedBox(height: 18),

              _ChartCard(
                title: isByBin
                    ? '${_selectedBinLabel()} recycle trend'
                    : 'Total recycled by day',
                data: data,
                maxValue: safeMaxValue,
              ),

              const SizedBox(height: 18),

              GridView.count(
                crossAxisCount: 2,
                shrinkWrap: true,
                physics: const NeverScrollableScrollPhysics(),
                crossAxisSpacing: 12,
                mainAxisSpacing: 12,
                childAspectRatio: 1.9,
                children: [
                  _SummaryTile(
                    icon: Icons.emoji_events,
                    title: 'Best day',
                    value: bestDay?.label ?? '-',
                    subtitle: bestDay == null
                        ? '0 L'
                        : '${bestDay.liters.toStringAsFixed(1)} L',
                    color: const Color(0xFFEAF6C8),
                  ),
                  _SummaryTile(
                    icon: Icons.trending_up,
                    title: 'Average / day',
                    value: '${average.toStringAsFixed(1)} L',
                    subtitle: 'Active days only',
                    color: const Color(0xFFFFF5C8),
                  ),
                  _SummaryTile(
                    icon: Icons.event_available,
                    title: 'Active days',
                    value: '$activeDays days',
                    subtitle: 'Has recycled waste',
                    color: const Color(0xFFE8F2FF),
                  ),
                  _SummaryTile(
                    icon: Icons.pie_chart,
                    title:
                    isByBin ? 'Share of total' : 'vs previous',
                    value: isByBin
                        ? '${share.toStringAsFixed(1)}%'
                        : '${percentChange >= 0 ? '+' : ''}$percentChange%',
                    subtitle:
                    isByBin ? 'All bins' : 'Previous period',
                    color: const Color(0xFFEAF6C8),
                  ),
                ],
              ),

              const SizedBox(height: 16),

              Container(
                width: double.infinity,
                padding: const EdgeInsets.all(16),
                decoration: BoxDecoration(
                  color: softGreen,
                  borderRadius: BorderRadius.circular(20),
                ),
                child: Row(
                  children: [
                    const Icon(
                      Icons.eco,
                      color: darkGreen,
                      size: 32,
                    ),
                    const SizedBox(width: 12),
                    Expanded(
                      child: Text(
                        isByBin
                            ? 'Nice! ${_selectedBinLabel()} contributed ${share.toStringAsFixed(1)}% of total recycled waste in this period.'
                            : percentChange >= 0
                            ? 'Great job! Recycled waste increased compared with the previous period.'
                            : 'Recycled waste decreased compared with the previous period. Keep improving!',
                        style: const TextStyle(
                          fontSize: 14,
                          fontWeight: FontWeight.w800,
                          height: 1.3,
                        ),
                      ),
                    ),
                  ],
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}

class _PeriodChip extends StatelessWidget {
  const _PeriodChip({
    required this.label,
    required this.selected,
    required this.onTap,
  });

  final String label;
  final bool selected;
  final VoidCallback onTap;

  static const darkGreen = Color(0xFF0B5D1E);

  @override
  Widget build(BuildContext context) {
    return Expanded(
      child: InkWell(
        borderRadius: BorderRadius.circular(22),
        onTap: onTap,
        child: Container(
          height: 42,
          alignment: Alignment.center,
          decoration: BoxDecoration(
            color: selected ? darkGreen : Colors.white,
            borderRadius: BorderRadius.circular(22),
            boxShadow: [
              BoxShadow(
                color: Colors.black.withOpacity(0.05),
                blurRadius: 10,
                offset: const Offset(0, 5),
              ),
            ],
          ),
          child: Text(
            label,
            style: TextStyle(
              fontWeight: FontWeight.w800,
              color: selected ? Colors.white : Colors.black87,
            ),
          ),
        ),
      ),
    );
  }
}

class _SegmentButton extends StatelessWidget {
  const _SegmentButton({
    required this.leftText,
    required this.rightText,
    required this.selectedIndex,
    required this.onChanged,
  });

  final String leftText;
  final String rightText;
  final int selectedIndex;
  final ValueChanged<int> onChanged;

  static const darkGreen = Color(0xFF0B5D1E);

  @override
  Widget build(BuildContext context) {
    return Container(
      height: 44,
      padding: const EdgeInsets.all(4),
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(22),
      ),
      child: Row(
        children: [
          Expanded(
            child: _SegmentItem(
              text: leftText,
              selected: selectedIndex == 0,
              onTap: () => onChanged(0),
            ),
          ),
          Expanded(
            child: _SegmentItem(
              text: rightText,
              selected: selectedIndex == 1,
              onTap: () => onChanged(1),
            ),
          ),
        ],
      ),
    );
  }
}

class _SegmentItem extends StatelessWidget {
  const _SegmentItem({
    required this.text,
    required this.selected,
    required this.onTap,
  });

  final String text;
  final bool selected;
  final VoidCallback onTap;

  static const darkGreen = Color(0xFF0B5D1E);

  @override
  Widget build(BuildContext context) {
    return InkWell(
      borderRadius: BorderRadius.circular(18),
      onTap: onTap,
      child: Container(
        alignment: Alignment.center,
        decoration: BoxDecoration(
          color: selected ? darkGreen : Colors.transparent,
          borderRadius: BorderRadius.circular(18),
        ),
        child: Text(
          text,
          style: TextStyle(
            color: selected ? Colors.white : Colors.black87,
            fontWeight: FontWeight.w800,
          ),
        ),
      ),
    );
  }
}

class _TotalCard extends StatelessWidget {
  const _TotalCard({
    required this.title,
    required this.total,
    required this.subtitle,
  });

  final String title;
  final double total;
  final String subtitle;

  static const darkGreen = Color(0xFF0B5D1E);

  @override
  Widget build(BuildContext context) {
    return Container(
      width: double.infinity,
      padding: const EdgeInsets.all(18),
      decoration: BoxDecoration(
        color: const Color(0xFFF3FAD7),
        borderRadius: BorderRadius.circular(22),
      ),
      child: Row(
        children: [
          Container(
            width: 62,
            height: 62,
            decoration: BoxDecoration(
              color: Colors.white.withOpacity(0.75),
              shape: BoxShape.circle,
            ),
            child: const Icon(
              Icons.recycling,
              color: darkGreen,
              size: 36,
            ),
          ),
          const SizedBox(width: 16),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  title,
                  style: const TextStyle(
                    fontSize: 15,
                    fontWeight: FontWeight.w900,
                  ),
                ),
                const SizedBox(height: 8),
                Row(
                  crossAxisAlignment: CrossAxisAlignment.end,
                  children: [
                    Text(
                      total.toStringAsFixed(1),
                      style: const TextStyle(
                        fontSize: 38,
                        fontWeight: FontWeight.w900,
                        height: 1,
                      ),
                    ),
                    const SizedBox(width: 6),
                    const Padding(
                      padding: EdgeInsets.only(bottom: 5),
                      child: Text(
                        'L',
                        style: TextStyle(
                          fontSize: 18,
                          fontWeight: FontWeight.w800,
                        ),
                      ),
                    ),
                  ],
                ),
                const SizedBox(height: 6),
                Row(
                  children: [
                    const Icon(Icons.trending_up, color: darkGreen, size: 17),
                    const SizedBox(width: 4),
                    Expanded(
                      child: Text(
                        subtitle,
                        style: const TextStyle(
                          color: darkGreen,
                          fontWeight: FontWeight.w800,
                        ),
                      ),
                    ),
                  ],
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }
}

class _ChartCard extends StatelessWidget {
  const _ChartCard({
    required this.title,
    required this.data,
    required this.maxValue,
  });

  final String title;
  final List<_RecycleDay> data;
  final double maxValue;

  @override
  Widget build(BuildContext context) {
    final safeMaxValue = maxValue <= 0 ? 1.0 : maxValue;

    return Container(
      height: 295,
      width: double.infinity,
      padding: const EdgeInsets.fromLTRB(16, 16, 16, 14),
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(22),
        boxShadow: [
          BoxShadow(
            color: Colors.black.withOpacity(0.06),
            blurRadius: 16,
            offset: const Offset(0, 7),
          ),
        ],
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              const Icon(Icons.bar_chart, color: Color(0xFF2F6B3D)),
              const SizedBox(width: 8),
              Expanded(
                child: Text(
                  title,
                  style: const TextStyle(
                    fontSize: 17,
                    fontWeight: FontWeight.w900,
                  ),
                ),
              ),
              Container(
                padding:
                const EdgeInsets.symmetric(horizontal: 10, vertical: 5),
                decoration: BoxDecoration(
                  borderRadius: BorderRadius.circular(16),
                  border: Border.all(color: Colors.black.withOpacity(0.1)),
                ),
                child: const Text(
                  'L',
                  style: TextStyle(fontWeight: FontWeight.w800),
                ),
              ),
            ],
          ),
          const SizedBox(height: 12),
          Expanded(
            child: data.isEmpty
                ? const Center(
              child: Text(
                'No recycle data in this period.',
                style: TextStyle(
                  color: Colors.black54,
                  fontWeight: FontWeight.w700,
                ),
              ),
            )
                : Row(
              crossAxisAlignment: CrossAxisAlignment.end,
              children: data.map((item) {
                final h = 155 * (item.liters / safeMaxValue);

                return Expanded(
                  child: Column(
                    mainAxisAlignment: MainAxisAlignment.end,
                    children: [
                      Text(
                        item.liters.toStringAsFixed(1),
                        style: const TextStyle(
                          fontSize: 12,
                          fontWeight: FontWeight.w800,
                        ),
                      ),
                      const SizedBox(height: 6),
                      Container(
                        width: 27,
                        height: h,
                        decoration: BoxDecoration(
                          color: const Color(0xFF69A83D),
                          borderRadius: BorderRadius.circular(0),
                        ),
                      ),
                      const SizedBox(height: 8),
                      Text(
                        item.label,
                        style: const TextStyle(
                          fontSize: 12,
                          fontWeight: FontWeight.w700,
                          color: Colors.black54,
                        ),
                      ),
                    ],
                  ),
                );
              }).toList(),
            ),
          ),
        ],
      ),
    );
  }
}

class _SummaryTile extends StatelessWidget {
  const _SummaryTile({
    required this.icon,
    required this.title,
    required this.value,
    required this.subtitle,
    required this.color,
  });

  final IconData icon;
  final String title;
  final String value;
  final String subtitle;
  final Color color;

  static const darkGreen = Color(0xFF0B5D1E);

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.all(13),
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(18),
        boxShadow: [
          BoxShadow(
            color: Colors.black.withOpacity(0.045),
            blurRadius: 12,
            offset: const Offset(0, 6),
          ),
        ],
      ),
      child: Row(
        children: [
          Container(
            width: 42,
            height: 42,
            decoration: BoxDecoration(
              color: color,
              shape: BoxShape.circle,
            ),
            child: Icon(icon, color: darkGreen, size: 22),
          ),
          const SizedBox(width: 9),
          Expanded(
            child: Column(
              mainAxisAlignment: MainAxisAlignment.center,
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  title,
                  maxLines: 1,
                  overflow: TextOverflow.ellipsis,
                  style: const TextStyle(
                    fontSize: 11.5,
                    color: Colors.black54,
                    fontWeight: FontWeight.w800,
                  ),
                ),
                const SizedBox(height: 3),
                Text(
                  value,
                  maxLines: 1,
                  overflow: TextOverflow.ellipsis,
                  style: const TextStyle(
                    fontSize: 15,
                    fontWeight: FontWeight.w900,
                  ),
                ),
                Text(
                  subtitle,
                  maxLines: 1,
                  overflow: TextOverflow.ellipsis,
                  style: const TextStyle(
                    fontSize: 11.5,
                    color: darkGreen,
                    fontWeight: FontWeight.w800,
                  ),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }
}

class _RecycleStats {
  final String binId;
  final String startDate;
  final String endDate;
  final double totalLiters;
  final double previousTotalLiters;
  final int percentChange;
  final List<_RecycleDay> days;

  const _RecycleStats({
    required this.binId,
    required this.startDate,
    required this.endDate,
    required this.totalLiters,
    required this.previousTotalLiters,
    required this.percentChange,
    required this.days,
  });

  factory _RecycleStats.fromJson(Map<String, dynamic> json) {
    final rawDays = json['days'];

    final days = rawDays is List
        ? rawDays.map((item) {
      final map = Map<String, dynamic>.from(item as Map);

      return _RecycleDay(
        (map['label'] ?? '').toString(),
        _toDouble(map['liters']),
      );
    }).toList()
        : <_RecycleDay>[];

    return _RecycleStats(
      binId: (json['binId'] ?? '').toString(),
      startDate: (json['startDate'] ?? '').toString(),
      endDate: (json['endDate'] ?? '').toString(),
      totalLiters: _toDouble(json['totalLiters']),
      previousTotalLiters: _toDouble(json['previousTotalLiters']),
      percentChange: _toInt(json['percentChange']),
      days: days,
    );
  }

  static double _toDouble(dynamic value) {
    if (value == null) return 0.0;
    if (value is int) return value.toDouble();
    if (value is double) return value;
    if (value is num) return value.toDouble();
    return double.tryParse(value.toString()) ?? 0.0;
  }

  static int _toInt(dynamic value) {
    if (value == null) return 0;
    if (value is int) return value;
    if (value is double) return value.round();
    if (value is num) return value.round();
    return int.tryParse(value.toString()) ?? 0;
  }
}

class _RecycleDay {
  final String label;
  final double liters;

  const _RecycleDay(this.label, this.liters);
}

class _BinOption {
  final String id;
  final String name;
  final String location;

  const _BinOption({
    required this.id,
    required this.name,
    required this.location,
  });
}

class _DateRange {
  final DateTime start;
  final DateTime end;

  const _DateRange({
    required this.start,
    required this.end,
  });
}