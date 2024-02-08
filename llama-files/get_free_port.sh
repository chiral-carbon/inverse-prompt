#!/bin/bash

for port in $(shuf -i 30000-65500 -n 40); do
	if [[ $(netstat -tupln 2>&1 | grep $port | wc -l) -eq 0 ]]; then
		echo $port
		#free_ports="${free_ports} $port"
		break
	fi
done
