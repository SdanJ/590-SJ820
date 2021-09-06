ScriptLoc=${PWD}
cd LectureCodes
for i in *.py; do echo $i; python $i; done
cd $ScriptLoc
for i in *.py; do echo $i; python $i; done
