#!/bin/bash

if [ $1 == "blanca" ]
then
    sshPATH=cosi1728@login.rc.colorado.edu:/projects/cosi1728/Implicit-Neural-Compression
elif [ $1 == "hyak" ]
then
    sshPATH=rscooper@klone.hyak.uw.edu:projects/Implicit-Neural-Compression
fi

if [ $2 == "scp" ]
then
    scp -r $sshPATH/lightning_logs/$3 lightning_logs/$4
elif [ $2 = "rsync" ]
then
    rsync -rP $sshPATH/lightning_logs/$3 lightning_logs/$4
fi
