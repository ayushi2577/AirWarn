# Data summary: city_day.csv

## 1. Basic facts
- Rows: **29,531**, columns: **16**
- Columns: City, Date, PM2.5, PM10, NO, NO2, NOx, NH3, CO, SO2, O3, Benzene, Toluene, Xylene, AQI, AQI_Bucket
- Date range: **2015-01-01 to 2020-07-01**
- Cities: **26** (Ahmedabad, Aizawl, Amaravati, Amritsar, Bengaluru, Bhopal, Brajrajnagar, Chandigarh, Chennai, Coimbatore, Delhi, Ernakulam, Gurugram, Guwahati, Hyderabad, Jaipur, Jorapokhar, Kochi, Kolkata, Lucknow, Mumbai, Patna, Shillong, Talcher, Thiruvananthapuram, Visakhapatnam)
- Duplicate city-date rows: **0**

## 2. Coverage per city (first date, last date, rows)
```
                        first       last  rows  calendar_days  missing_days  aqi_present
City                                                                                    
Ahmedabad          2015-01-01 2020-07-01  2009           2009             0         1334
Mumbai             2015-01-01 2020-07-01  2009           2009             0          775
Lucknow            2015-01-01 2020-07-01  2009           2009             0         1893
Bengaluru          2015-01-01 2020-07-01  2009           2009             0         1910
Delhi              2015-01-01 2020-07-01  2009           2009             0         1999
Chennai            2015-01-01 2020-07-01  2009           2009             0         1884
Hyderabad          2015-01-04 2020-07-01  2006           2006             0         1880
Patna              2015-06-01 2020-07-01  1858           1858             0         1459
Gurugram           2015-11-27 2020-07-01  1679           1679             0         1453
Visakhapatnam      2016-07-01 2020-07-01  1462           1462             0         1171
Amritsar           2017-02-27 2020-07-01  1221           1221             0         1126
Jorapokhar         2017-04-20 2020-07-01  1169           1169             0          771
Jaipur             2017-06-14 2020-07-01  1114           1114             0         1094
Thiruvananthapuram 2017-06-16 2020-07-01  1112           1112             0         1052
Amaravati          2017-11-24 2020-07-01   951            951             0          841
Brajrajnagar       2017-12-07 2020-07-01   938            938             0          713
Talcher            2017-12-20 2020-07-01   925            925             0          698
Kolkata            2018-04-10 2020-07-01   814            814             0          754
Guwahati           2019-02-16 2020-07-01   502            502             0          495
Coimbatore         2019-06-12 2020-07-01   386            386             0          344
Shillong           2019-08-27 2020-07-01   310            310             0          205
Chandigarh         2019-09-02 2020-07-01   304            304             0          299
Bhopal             2019-09-17 2020-07-01   289            289             0          278
Ernakulam          2020-01-22 2020-07-01   162            162             0          153
Kochi              2020-01-22 2020-07-01   162            162             0          158
Aizawl             2020-03-11 2020-07-01   113            113             0          111
```

## 3. Missing values (% of rows)
```
Xylene        61.3
PM10          37.7
NH3           35.0
Toluene       27.2
Benzene       19.0
AQI           15.9
AQI_Bucket    15.9
PM2.5         15.6
NOx           14.2
O3            13.6
SO2           13.1
NO2           12.1
NO            12.1
CO             7.0
```
- Pollutants never measured in some cities:
  - PM10: Lucknow
  - NOx: Jorapokhar
  - NH3: Ahmedabad
  - O3: Ernakulam
  - Benzene: Bhopal, Jorapokhar, Kochi, Thiruvananthapuram
  - Toluene: Bhopal, Brajrajnagar, Guwahati, Jorapokhar, Thiruvananthapuram
  - Xylene: Aizawl, Bengaluru, Bhopal, Brajrajnagar, Chennai, Coimbatore, Guwahati, Jaipur, Jorapokhar, Kochi, Lucknow, Shillong, Talcher, Thiruvananthapuram

## 4. Value checks
- Negative values: none
```
          min    50%    99%     max
PM2.5     0.0   48.6  311.1   950.0
PM10      0.0   95.7  444.6  1000.0
NO        0.0    9.9  116.3   390.7
NO2       0.0   21.7  117.1   362.2
NOx       0.0   23.5  154.3   467.6
NH3       0.0   15.8  130.6   352.9
CO        0.0    0.9   34.5   175.8
SO2       0.0    9.2  102.7   193.9
O3        0.0   30.8  105.3   257.7
Benzene   0.0    1.1   28.7   455.0
Toluene   0.0    3.0   72.5   454.8
Xylene    0.0    1.0   26.7   170.4
AQI      13.0  118.0  661.5  2049.0
```

## 5. Does AQI_Bucket match the AQI value?
- Rows with both: 24,850; mismatches: **0**
- Bucket counts: {'Moderate': 8829, 'Satisfactory': 8224, 'Poor': 2781, 'Very Poor': 2337, 'Good': 1341, 'Severe': 1338}

## 6. Target: poor_tomorrow = 1 if AQI on day t+1 > 200
- Rows whose next calendar day is absent from the file: 26
- Rows whose next day exists but has no AQI: 4,656
- Rows with tomorrow's AQI (target defined): 24,849
- Of those, today's AQI is blank on **506** rows -> excluded by the evaluation-rows rule
- **Evaluation rows (today's and tomorrow's AQI both observed): 24,343**
- **Positives: 6,330 (26.0%)**
- Persistence check: tomorrow's yes/no status equals today's on **92.8%** of evaluation rows (this is the persistence baseline's accuracy)
- Change days (binary flip, <=200 vs >200): **1,748** (clean->bad: 868, bad->clean: 880)

### Positive rate by year
```
      rows  positives   rate
Year                        
2015  1802        750  0.416
2016  2486        929  0.374
2017  3152       1039  0.330
2018  5601       1593  0.284
2019  6968       1576  0.226
2020  4334        443  0.102
```
### Positive rate by city
```
                    rows  positives   rate
City                                      
Ahmedabad           1298       1063  0.819
Delhi               1993       1300  0.652
Patna               1409        753  0.534
Gurugram            1401        726  0.518
Lucknow             1879        931  0.495
Talcher              659        194  0.294
Guwahati             490        132  0.269
Kolkata              752        198  0.263
Jorapokhar           735        167  0.227
Brajrajnagar         672        149  0.222
Bhopal               276         35  0.127
Jaipur              1085        122  0.112
Amritsar            1095        117  0.107
Visakhapatnam       1124         89  0.079
Chandigarh           294         23  0.078
Amaravati            798         54  0.068
Chennai             1872        123  0.066
Mumbai               769         36  0.047
Hyderabad           1867         71  0.038
Bengaluru           1901         41  0.022
Kochi                155          2  0.013
Thiruvananthapuram  1040          4  0.004
Coimbatore           324          0  0.000
Aizawl               109          0  0.000
Ernakulam            148          0  0.000
Shillong             198          0  0.000
```

## 7. Split and fold sizes (embargo days already excluded)
```
                     part       from         to  rows  positives  rate  cities
        train (2015-2018) 2015-01-01 2018-12-30 13024       4300 0.330      18
              test (2019) 2019-01-01 2019-12-30  6946       1567 0.226      23
              seen cities 2019-01-01 2019-12-30  6184       1453 0.235      18
               new cities 2019-01-01 2019-12-30   762        114 0.150       5
stress: 2020 pre-lockdown 2020-01-01 2020-03-23  1941        409 0.211      26
              seen cities 2020-01-01 2020-03-23  1423        344 0.242      18
               new cities 2020-01-01 2020-03-23   518         65 0.125       8
    stress: 2020 lockdown 2020-03-25 2020-12-31  2369         32 0.014      26
              seen cities 2020-03-25 2020-12-31  1667         22 0.013      18
               new cities 2020-03-25 2020-12-31   702         10 0.014       8
             fold 1 train 2015-01-01 2016-12-30  4280       1675 0.391       9
          fold 1 validate 2017-01-01 2017-06-30  1396        506 0.362      12
             fold 2 train 2015-01-01 2017-06-29  5676       2185 0.385      13
          fold 2 validate 2017-07-01 2017-12-31  1756        533 0.304      15
             fold 3 train 2015-01-01 2017-12-30  7427       2711 0.365      15
          fold 3 validate 2018-01-01 2018-06-30  2619        797 0.304      18
             fold 4 train 2015-01-01 2018-06-29 10042       3514 0.350      18
          fold 4 validate 2018-07-01 2018-12-30  2965        785 0.265      18
```
- Cities never in training (8): Aizawl, Bhopal, Chandigarh, Coimbatore, Ernakulam, Guwahati, Kochi, Shillong
- New-city groups with fewer than 50 positives should be reported but flagged as low-reliability.

## 8. Most correlated pairs (|r| > 0.7)
```
PM2.5    PM10       0.85
PM10     AQI        0.80
NO       NOx        0.79
Benzene  Toluene    0.74
```
