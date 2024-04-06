# WELCOME TO SCREEN SIGHT!

Hello and welcome to Screen Sight, the tool designed to address the
current limitations of passthrough mode in modern virtual reality (VR) headsets.

This document will cover the installation and use of Screen Sight.
Please note that this tool is still under development, and this version is simply
a proof of concept demo.

Also note that the demo currently only works with Android devices.

<i>Created by:</i>

- Lucas Hartman (<a href='mailto: lhartma8@uwo.ca'>lhartma8@uwo.ca</a>),
- Dhruv Sagre (<a href='mailto: dsagre@uwo.ca'>dsagre@uwo.ca</a>),
- Ethan Pigou (<a href='mailto: epigou@uwo.ca'>epigou@uwo.ca</a>)

## 1. Installation Instructions

To use Screen Sight, there are 4 things you need to have:
1. The Source Code: Clone the repository to download the files needed to run Screen Sight
2. Scrcpy: A tool used to screen mirror on Android phones
3. SensorServer: A tool used to gather sensor data on Android phones
4. Libraries: There are several python libraries required to run Screen Sight

### 1.a) ScrcPy

ScrcPy is a tool which allows you to mirror your Android phone's screen
on a computer. A copy of it is included within this repository in the
'/scrcpy' directory.

Once the repository has been cloned to your computer, get the absolute path
of the /scrcpy directory. Depending on where you install the files, it will
look something like this:

>C:\Users\<your_username>\Downloads\Screen Sight\scrcpy

Copy this path and go to:

>Control Panel > View advanced system settings > Environment Variables

Find the system variable that says "Path", click on it, and click the edit button.
Click the "New" button, and paste in the copied path from above. Then click "OK"
and close out of the windows.

Next, you need to setup your Android phone to be compatible with Scrcpy. To do
this, go into your phone's settings. Navigate to "About Phone" and scroll to the
bottom until you see the "Build number" property. Click on this 7 times repeatedly
to enable developer mode.

Now go back to your general settings, and navigate to "Developer options". Scroll
down until you see "USB debugging". Make sure this is toggled on.

Now you just need to actually run the tool. Make sure your phone is plugged
into your computer via USB. Open a terminal, and run:

>adb devices

This should print out "List of devices attached" and then your device. If you
don't see this, you may need to do some troubleshooting.

By running the below code, you should open a window on your computer that is
a mirror of your screen:

>scrcpy --window-title "SCRCPY"

If you want to use scrcpy wirelessly, you just need to find your phone's
local IP address and run the following commands:

>adb tcpip 5555
> 
>adb connect <your_ip>:5555

Then run the below command to open the mirrored window again:

>scrcpy --window-title "SCRCPY"

Credit to Genymobile for creating this tool:
https://github.com/Genymobile/scrcpy

### 1.b) SensorServer

Sensor server is the tool that allows you to get the accelerometer data
from your Android phone. A copy of this tool is included within this
repository for convenience. It is the file: 'SensorServer-v5.3.0.apk'.

To install this app, you first need to enable your phone to allow installing
unknown apps. Make sure you have developer mode enabled on your android phone,
which is explained in the ScrcPy installation tutorial above.

Go to your phone's settings, and navigate to:

> Special app access > Install unknown apps

Enable whichever app you wish to use to store the .apk file. Now use that app to
send the 'SensorServer-v5.3.0.apk' file from this repository to your phone. Open
the .apk file and allow the installation to take place.

Open the app, and click the "Start" button.

You will now see a path which is the socket opened by your phone. It will look
something like this:

>ws://<your_ip>:8080

Copy this entire string. Now go to the 'app.py' file in this
directory, and paste this string in the phone_url variable.

Credit to umer0586 for creating this tool:
https://github.com/umer0586/SensorServer

### 1.c) Required libraries

To install the required libraries, simply run the following command in the root
of this directory:

> pip install -r requirements.txt

This will install the CPU version of PyTorch, but it is reccomended to use
the GPU version with CUDA for the best performance.


## 2. Using Screen Sight

To use Screen Sight, first make sure you have completed the installation
steps above.

Specifically, make sure you ahve collected the phone URL from SensorServer
and have opened a screen mirror on your computer using:

>scrcpy --window-title "SCRCPY"

Now, simply run the following command in the root of this directory:

>python app.py

Make sure your SCRCPY window is not minimized or behind any other windows while
running this app. Also make sure you do not lose connection to SensorServer.