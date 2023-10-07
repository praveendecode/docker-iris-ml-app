FROM python:3.10
WORKDIR /app
COPY Regression_Model.py .
COPY Iris_Data.csv .
RUN pip install pandas scikit-learn 
CMD python Regression_Model.py
