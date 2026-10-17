# OCR Web Application

## Goal

A web application where a user uploads an image or PDF document.
The server sends the document to an OCR model and returns the
transcribed text to the user.

### First version

Input:
 - JPG
 - JPEG
 - PNG
 - PDF

 Output:
  - Plain text

First version does NOT include:
 - authentication
 - user accounts
 - database
 - payment
 - history
 - cloud storage
 - multiple users
 - job queues

Live demo: https://flame-tigress-unmovable.ngrok-free.dev
Upload a JPG/PNG/PDF (≤ 20 MB) and get the text back. It runs Chandra OCR 2 on my own GPU server. (If you see an ngrok notice, click "Visit Site".)