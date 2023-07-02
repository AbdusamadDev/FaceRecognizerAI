from rest_framework.response import Response
from rest_framework.generics import CreateAPIView
from rest_framework.authtoken.views import ObtainAuthToken
from rest_framework.authtoken.models import Token
from rest_framework.views import APIView
from django.contrib.auth import authenticate, login, logout
from django.urls import reverse_lazy
from django.http import StreamingHttpResponse
from django.conf import settings
import zipfile
import shutil
import os
import cv2

import face_pro_3
from base.models import Employee
from base.serializers import EmployeeSerializer


def home(request):
    return Response("Hello Web Camera APP!!!!!")


class CreateEmployeeView(CreateAPIView):
    serializer = EmployeeSerializer
    model = Employee
    success_url = reverse_lazy("home")

    def post(self, request, *args, **kwargs):
        serializer = self.serializer(data=request.data)
        if serializer.is_valid(raise_exception=True):
            uploaded_zip_file = request.FILES.get('folder')
            id = serializer.validated_data.get("employee_id")
            path = f"media/{id}"

            if uploaded_zip_file is not None:
                zip_file_path = os.path.join(settings.MEDIA_ROOT, f'{id}.zip')

                if not os.path.exists(os.path.dirname(zip_file_path)):
                    os.makedirs(os.path.dirname(zip_file_path))

                with open(zip_file_path, 'wb+') as destination:
                    for chunk in uploaded_zip_file.chunks():
                        destination.write(chunk)

                with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
                    zip_ref.extractall(path)
                    for index, file in enumerate(os.listdir(path), start=1):
                        old_file_path = os.path.join(path, file)
                        new_file_path = os.path.join(path, f"{index}.jpg")
                        os.rename(old_file_path, new_file_path)

                os.remove(zip_file_path)

            main_image = request.FILES.get("image")
            if main_image is not None:
                main_image_path = os.path.join(settings.MEDIA_ROOT, 'main.jpg')
                with open(main_image_path, 'wb+') as destination:
                    for chunk in main_image.chunks():
                        destination.write(chunk)
                shutil.move(main_image_path, path)
                serializer.image = f"{id}/main.jpg"

            serializer.validated_data.pop("folder", None)
            serializer.validated_data.pop("image", None)
            serializer.save()
            print(request.user.is_authenticated)
            return Response({"msg": "Data created!!!"}, status=201)


class LiveStreamAPIView(APIView):
    def get(self, request):
        cap = cv2.VideoCapture(0)
        emotion_model = face_pro_3.create_emotion_model()
        emotion_model.load_weights('facial_expression_model_weights.h5')

        image_files = [
            ["sanjar01.jpg", "sanjar02.jpg", "sanjar03.jpg"],
            ["diyor01.jpg", "diyor02.jpg", "diyor03.jpg"],
            ["abdusamad01.jpg"],
            ["javohir01.jpg", "javohir02.jpg"]
        ]
        names = ["Sanjar", "Diyor", "Abdusamad", "Javohir"]

        known_face_encodings, known_face_names = face_pro_3.load_face_encodings(image_files, names)

        video_capture = cv2.VideoCapture(0)
        function = face_pro_3.detect_and_display_faces(
            video_capture, known_face_encodings, known_face_names, emotion_model)

        # while True:
        #     ret, frame = cap.read()
        #     if not ret:
        #         break
        #     else:
        #         frame = cv2.flip(frame, 1)
        #         ret, buffer = cv2.imencode('.jpg', frame)
        #         frame = buffer.tobytes()
        #         yield (
        #                 b'--frame\r\n'
        #                 b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n'
        #         )

        cap.release()

        return StreamingHttpResponse(function, content_type='multipart/x-mixed-replace; boundary=frame')


class GetAuthTokenAPIView(ObtainAuthToken):
    def post(self, request, *args, **kwargs):
        serializer = self.serializer_class(
            data=request.data,
            context={'request': request}
        )
        if request.user.is_authenticated:
            if serializer.is_valid(raise_exception=True):
                user = serializer.validated_data['user']
                token, created = Token.objects.get_or_create(user=user)
                return Response({'token': token.key}, status=200)
        else:
            user = authenticate(
                request=request,
                username=request.data.get("username"),
                password=request.data.get("password")
            )
            if user:
                print("User is available")
                login(request, user=user)
            serializer.is_valid(raise_exception=True)
            user = serializer.validated_data['user']
            token, created = Token.objects.get_or_create(user=user)
            return Response({'token': token.key}, status=200)
