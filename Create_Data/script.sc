#!/bin/bash
# This is a Bash file to decode the Power of the .journal files to CSV

echo 'executing the .elf file'

path_of_elf="../decoders/wsn_log_decoder_v3.elf"
path_of_file="../CDSA_JOURNAL"
year=("2017" "2018" "2019" "2020")
month=("01" "02" "03" "04" "05" "06" "07" "08" "09" "10" "11" "12")
day=("01" "02" "03" "04" "05" "06" "07" "08" "09" "10" "11" "12" "13" "14" "15" "16" "17" "18" "19" "20" "21" "22" "23" "24" "25" "26" "27" "28" "29" "30" "31")
dest_file="powers_${year[0]}_${month[8]}.txt"

echo "Path of the .elf file: $path_of_elf"
echo "Current year: ${year[0]}"
echo "Current month: "${month[8]}
echo "File to save $dest_file"

$path_of_elf -f "$path_of_file"_${year[0]}/Journal/${year[0]}/${month[8]}/traceback_2017-09-06.journal -c 0 -d csv > $dest_file

echo "executed with number of lines $(wc $dest_file -l)"

for ((y=0; y<=3; y++))
do
	mkdir ${year[y]} -v

	m=0
	if [ $y -gt 0 ]
	then
		m=0
	else
		m=8
	fi

	for ((; m<=11; m++))
	do
		mkdir ${year[y]}/${month[m]}
		for ((d=0; d<=30; d++))
		do
			destination_file="${year[y]}/${month[m]}/powers_${year[y]}_${month[m]}_${day[d]}.txt"
			$path_of_elf -f "$path_of_file"_${year[y]}/Journal/${year[y]}/${month[m]}/traceback_${year[y]}-${month[m]}-${day[d]}.journal -d csv > $destination_file
		done
	echo "${year[y]}/${month[m]} with $(ls ${year[y]}/${month[m]}/ | wc -l) files created"
	done
done

echo "DONE"

