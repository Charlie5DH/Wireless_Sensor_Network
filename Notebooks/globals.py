# defining some constants
Type=['Network Coordinator', 'Radio - 2.4 GHz','Acquisition - Temperature',
      'Acquisition - Current / Voltage', 'Power - Solar Panel', 'Power - AC/DC Input']

Modules={"00.57.FE.04":'Net-Coordinator',
         "00.57.FE.0E":'Radio-2.4 GHz',
         "00.57.FE.0F":'Radio-2.4 GHz',
         "00.57.FE.06":'Radio-2.4 GHz',
         "00.57.FE.09":'Radio-2.4 GHz',
         "00.57.FE.01":'Radio-2.4 GHz',
         "00.57.FE.05":'Radio-2.4 GHz',
         "00.57.FE.03":'Radio-2.4 GHz',
         "29.E5.5A.24":'Acq-Tempe',
         "A7.CB.0A.C0":'Acq-Current/Volt',
         "34.B2.9F.A9":'P-Solar Panel',
         "01.E9.39.32":'Acq-Current/Volt',
         "A4.0D.82.38":'P-AC/DC Input',
         "9F.8D.AC.91":'Acq-Tempe',
         "50.39.E2.80":'P-Solar Panel'}

column_names=['Timestamp','Module','Type','Temp_Mod', 'VBus',
              'PT100(0)', 'PT100(1)', 'LVL_Dim(1)', 'V_MPPT',
              'V_Panel','LVL_Drain(1)','VBat', 'V_Supp','Temp_Oil',
              'Temp_gab','V_MPPT_TE','V_Panel_TE']

column_names_dates = ['Timestamp', 'Module', 'Type', 'month',
       'year', 'day', 'week', 'hour', 'Temp_Mod', 'VBus', 'PT100(0)',
       'PT100(1)', 'LVL_Dim(1)', 'V_MPPT', 'V_Panel', 'LVL_Drain(1)', 'VBat',
       'V_Supp', 'Temp_Oil', 'Temp_gab', 'V_MPPT_TE', 'V_Panel_TE']

columns_radio = ['Timestamp','Module','Type', 'Transmitter', 'N1', 'P_N1(dbm)',
                 'N2', 'P_N2(dbm)', 'N3', 'P_N3(dbm)',
                 'N4','P_N4(dbm)', 'N5', 'P_N5(dbm)',
                 'N6', 'P_N6(dbm)', 'N7','P_N7(dbm)']

columns_radio2 = ['Timestamp','Module','Type', 'Receiver', 'Tx1', 'P_Tx1(dbm)',
                 'Tx2', 'P_Tx2(dbm)', 'Tx3', 'P_Tx3(dbm)',
                 'Tx4','P_Tx4(dbm)', 'Tx5', 'P_Tx5(dbm)',
                 'Tx6', 'P_Tx6(dbm)', 'Tx7','P_Tx7(dbm)']
