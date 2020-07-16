#!/bin/sh

case "$1" in
    pull )
	eval "$(ssh-agent -s)"
	ssh-add	
	git pull
	;;
    push )
	if [ "$#" != 2 ]
	then 
	    echo "[info] 2 args needed, you may forget push message."
	    exit 0
	fi

	eval "$(ssh-agent -s)"
	ssh-add	
	git add . 
	git commit -m "$2"
	git push -u origin master
	;;	
    ssh )
	eval "$(ssh-agent -s)"
	ssh-add	
	;; 
    * )
    	echo "[info]Nothing changed."
esac

exit 0
