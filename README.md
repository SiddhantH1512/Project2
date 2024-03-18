# Banking Analytics Project

This banking project is an educational endeavor that features a comprehensive analysis framework for a fictional bank with branches in Spain, Germany, and France. The project is segmented into three pivotal modules focusing on various aspects of banking analytics:

1. **Fraud Detection**: Identifying and flagging fraudulent transactions to ensure the integrity and trustworthiness of banking operations.
2. **Loan Conversion Potential**: Analyzing depositor behaviors to pinpoint potential customers who might be interested in taking out loans, thereby facilitating targeted marketing strategies.
3. **Customer Churn Prediction**: Predicting the likelihood of customers discontinuing their services, enabling proactive measures to enhance customer retention.

The project harnesses the power of the Flask framework for backend operations and employs Data Version Control (DVC) for meticulous data management. Additionally, a robust CI/CD pipeline is in place, automating the processes of dockerization, pushing the Docker image to Amazon Elastic Container Registry (ECR), and deploying it on an Amazon EC2 instance.

## Getting Started

To clone this project for educational or developmental purposes, execute the following command:

```bash
git clone https://github.com/SiddhantH1512/Project2.git
```

## Prerequisites

Ensure you have the following tools installed:

- [Git](https://git-scm.com/downloads) - Version control system
- [Docker](https://docs.docker.com/get-docker/) - Container platform
- [Python](https://www.python.org/downloads/) - Programming language (with Flask)
- [DVC](https://dvc.org/) - Data Version Control for data and ML projects


## Installation
Follow these steps to set up your local development environment:
1. Clone the repository
```
git clone https://github.com/SiddhantH1512/Project2.git
```
2. Navigate to project directory
```
cd Project2
```
3. Install the necessary python packages
```
pip install -r requirements.txt
```

## Usage
Here's how you can run the project locally after installation:
1. Activate the Flask application
```
export FLASK_APP=app.py
flask run
```
2. Access the web application at http://localhost:8000


## Built With

- [Flask](https://flask.palletsprojects.com/en/2.0.x/) - The web framework used.
- [DVC](https://dvc.org/) - Data Version Control for handling large datasets and versioning.
- [GitHub Actions](https://github.com/features/actions) - CI/CD processes utilizing GitHub Actions for automated testing, building, and deployment.

## Authors

- [SiddhantH1512](https://github.com/SiddhantH1512) - Profile


## Acknowledgments
1. Data obtained from Kaggle for analytical insights.
2. Flask community for extensive documentation and support.


