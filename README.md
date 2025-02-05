<p align="center">
    <h3 align="center">CNN Model for Digit Prediction</h3>
  </a>
</p>

<p align="center">Python project to train dataset with CNN model for digit predcition. It exports tunned model to the backend of <a href="https://github.com/arnon3339/mnist-project.git">Mnits project</a>.</p>

<br/>

## Introduction

The CNN model has to be fine-tunned with hyperparameters. The [Optuna](https://optuna.org/) library is used to fine the best model for digit prediction.


## How it works

## Demo

https://mnist-project.vercel.app/

## Repositories

Deployment: https://github.com/arnon3339/mnist-project.git  
CNN model: https://github.com/arnon3339/mnist-model.git  
Docker deployment: https://github.com/arnon3339/mnist-model.git

## Deploy Your Own

You can clone & deploy it to Vercel with one click:
[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https%3A%2F%2Fgithub.com%2Farnon3339%2Fmnist-project%2Ftree%2Fmain)


## Developing Locally

You can clone & create this repo with the following command

```bash
npx create-next-app nextjs-fastapi --example "https://github.com/arnon3339/mnist-project.git"
```

## Getting Started

First, create and activate a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
```

Then, install the dependencies:

```bash
npm install
# or
yarn
# or
pnpm install
```

Then, run the development server(python dependencies will be installed automatically here):

```bash
npm run dev
# or
yarn dev
# or
pnpm dev
```

Open [http://localhost:3000](http://localhost:3000) with your browser to see the result.

The FastApi server will be running on [http://127.0.0.1:8000](http://127.0.0.1:8000) – feel free to change the port in `package.json` (you'll also need to update it in `next.config.js`).

## Dataset

The CNN model use mnist dataset that is the open source. It can be found in [Kraggle](https://www.kaggle.com). An image can be uploaded for digit prediction.

## Learn More

To learn more about Next.js, take a look at the following resources:

- [Next.js Documentation](https://nextjs.org/docs) - learn about Next.js features and API.
- [Learn Next.js](https://nextjs.org/learn) - an interactive Next.js tutorial.
- [FastAPI Documentation](https://fastapi.tiangolo.com/) - learn about FastAPI features and API.

You can check out [the Next.js GitHub repository](https://github.com/vercel/next.js/) - your feedback and contributions are welcome!