#!/bin/bash

try() {
	"$@"
	if [ "$?" != "0" ]; then
		echo "Command failed: $@"
		exit 1
	fi
}

try docker build . | tee build.out

IMID=`try cat build.out | grep 'Successfully built' | awk '{print $3}'`

echo "Image ID is $IMID"

# From here on, SILS specific stuff

try docker run -v $PWD/..:/sils -w /sils -t -d $IMID /bin/bash > cid.out

CID=`cat cid.out`

echo "Container ID is $CID"

echo "Installing SILS dependencies..."
try docker exec $CID /bin/bash packages.txt
try docker exec $CID pip3 install -r requirements.txt

echo "Setting up language stuff..."
try docker exec $CID python3 setup.py

echo "Running test..."
# Just in case there was a previous failed run
docker exec -t $CID rm -rf /tmp/sils-cache/*

try docker exec -t $CID python3 server/lib/icongenerator2.py

echo "Cleaning up..."
try docker kill $CID
try docker rm $CID
try docker rmi -f $IMID