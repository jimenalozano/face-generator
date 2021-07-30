# Face Generator
Face Generation Tool for Memory Experimentation

### Set up

Install virtualenv to keep all your dependencies in one place.
```bash
sudo apt-get install virtualenv
```

After that, create a virtual environment
```bash
virtualenv venv --python=python3.6
```
This should create a venv folder. Activate it by running
```bash
source venv/bin/activate
```
Install the dependencies
```bash
pip install -r requirements.txt
```
Check hardware requirements are OK
```bash
python cli/cli.py
```
Finally, you are ready to go! Run the face generator services from:
```bash
cd service
python service.py
```

Notice that, if you dont have the database, you should uncomment line 24 of service.py to create the table needed to store the seeds of every face generated. In that way you can later use the transition service, by choosing the images from the database.

You can see the results at face-generator/results/

Once you are all done, get out of the virtual environment by running
```bash
deactivate
```