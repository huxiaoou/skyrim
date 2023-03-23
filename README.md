# skyrim

This project is designed to provide some highly frequently used tools in daily research.

---

## whiterun.py

+ Class:CCalendar, to manage trade days in research, provide interfaces to get satisfy multiple needs.
+ Function:fix_contract_id, this function is designed to solve the bug caused by the naming rule of CZC. More information can be found in DataManager/README.md

## falkreath.py

+ Class:CManagerLibReader, a class to provide a more convenient interface to read data from local sqlite3 database.
+ Class:CManagerLibWriterByDate, a class to provide a more convenient interface to manage the market data. A local database is created by this class, to record some intermediary results.

