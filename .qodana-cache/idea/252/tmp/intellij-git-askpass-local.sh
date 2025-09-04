#!/bin/sh
"/opt/idea/jbr/bin/java" -cp "/opt/idea/plugins/vcs-git/lib/git4idea-rt.jar:/opt/idea/lib/externalProcess-rt.jar" git4idea.http.GitAskPassApp "$@"
