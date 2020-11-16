# WSN **Project**
Different files to create data using the extracted .journal files. 

The extracted data and the CSV file with all combined is in https://drive.google.com/file/d/1FrHvWn6LV07Cr1v8F4M5h3x2uOiuNQNC/view?usp=sharing. It is important to note that the system has been live since 2017-09-20, with no reset. The link drops that occurred in this period were recovered by the Zigbee PRO stack. The most stable period was from July / 2018, before that we noticed many problems. The UG3 and UG5 routers are powered by AC from the dam, so each time they do maintenance, these nodes fall off and take some others along. And the router diagram can be seen in the coordinators page.

The data is in .journal files. These files can be decoded using `.elf` files. The commands are the next ones:

- `./wsn_log_decoder_v3.elf`  = programa decodificador
- `-f ../cachoeira/oct1/traceback_2017-10-01.journal`   = arquivo de entrada
- `-m 9F.8D.AC.91`   = mostrar apenas este rádio
- `-c 0`  = mostrar apenas o canal 0
- `-d csv` = a saída será em formato CSV

This command extracts the data in **channel 0** of an specific radio **9F.8D.AC.91** in CSV form<br>
`./wsn_log_decoder_v3.elf -f ../cachoeira/oct1/traceback_2017-10-01.journal -m 9F.8D.AC.91 -c 0 -d csv > teste.txt`

**Extract the entire network data:**

`./wsn_log_decoder.elf -f Journal/2017/08/traceback_2017-08-22.journal csv > decoded_csv.txt`

**Verify communication between all neighbors of a radio:**

`./wsn_log_decoder_v3.elf -f traceback_2017-08-21.journal -d csv -m 0x0057FE06 > x0.txt`

**Verify communication between a radios and a specific neighbor:**

`./wsn_log_decoder_v3.elf -f traceback_2017-08-21.journal -d csv -m 0x0057FE06 -n 0x0057FDFF > x1.txt`

**Extract the power RSSI of the radios**

`./wsn_log_decoder_v3.elf -f ../cachoeira/oct1/traceback_2017-10-01.journal -d csv > teste.txt`
