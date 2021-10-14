ScriptLoc=${PWD} 
cd MNIST
for i in *.py; do echo $i; python $i; done 
cd $ScriptLoc
cd DOGS-AND-CATS
for i in *.py; do echo $i; python $i; done 
