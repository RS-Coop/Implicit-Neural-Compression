#!/bin/bash

sshPATH=rscooper@klone.hyak.uw.edu:projects/Implicit-Neural-Compression

if [ $2 == "scp" ]
then
    scp -r $sshPATH/lightning_logs/$3 lightning_logs/$4
elif [ $2 = "rsync" ]
then
    rsync -rP $sshPATH/lightning_logs/$3 lightning_logs/$4
fi
