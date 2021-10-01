ScriptLoc=${PWD}
cd $ScriptLoc
for i in *.py; do echo $i; python $i; done
