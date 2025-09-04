#!/bin/sh
"/opt/idea/jbr/bin/java" -cp "/opt/idea/lib/externalProcess-rt.jar" externalApp.nativessh.NativeSshAskPassApp "$@"
