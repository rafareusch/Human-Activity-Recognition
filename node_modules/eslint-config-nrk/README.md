# eslint-config-nrk [![Build Status](https://travis-ci.org/nrkno/eslint-config-nrk.svg?branch=master)](https://travis-ci.org/nrkno/eslint-config-nrk) [![Coverage Status](https://img.shields.io/coveralls/nrkno/eslint-config-nrk/master.svg?style=flat)](https://coveralls.io/github/nrkno/eslint-config-nrk?branch=master) [![npm version](https://img.shields.io/npm/v/eslint-config-nrk.svg?style=flat)](https://www.npmjs.com/package/eslint-config-nrk)

An open/shared JavaScript coding standard configuration for NRK.no. Uses [`eslint`](http://eslint.org/) 2.x. If you're on version 1.x of `eslint`, see the [1.x branch](https://github.com/nrkno/eslint-config-nrk/tree/1.X).

## Installation

Execute these commands in the root of your project:

```sh
npm install --save-dev eslint
npm install --save-dev eslint-config-nrk
```

### Usage with React
If you are using React, you also need to install [`eslint-plugin-react`](https://github.com/yannickcr/eslint-plugin-react) within your project:

```sh
npm install --save-dev eslint-plugin-react
```

If you use the ```*.jsx``` extension, tell `eslint` explicitly to lint those files as well:

```sh
eslint . --ext=.js,.jsx
```

## Configuration

`eslint` is configured by adding an `.eslintrc.json` file to the root of your project, where
you specify [`enviroment(s)`](http://eslint.org/docs/user-guide/configuring#specifying-environments) and extend the `nrk` config. The following setup will lint your ES6 codebase:

```json
{
  "env": {
    "browser": true
  },
  "extends": [
    "nrk"
  ]
}
```

For React-support, you need to add the `react` extension as well:

```json
{
  "env": {
    "browser": true
  },
  "extends": [
    "nrk",
    "nrk/rules/react"
  ]
}
```

For JSX-support, add the `jsx` extension:

```json
{
  "env": {
    "browser": true
  },
  "extends": [
    "nrk",
    "nrk/rules/react",
    "nrk/rules/jsx"
  ]
}
```

Please note that the default config is for version 15 of React. If you use an older version, please refer to the [`documentation`](https://github.com/yannickcr/eslint-plugin-react#configuration).

### Legacy ES5 projects

```json
{
  "env": {
    "browser": true
  },
  "extends": [
    "nrk/es5"
  ]
}
```

See the [ESlint config docs](http://eslint.org/docs/user-guide/configuring#extending-configuration-files) for more information.
