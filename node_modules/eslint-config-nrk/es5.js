module.exports = {
  env: {
    es6: false
  },
  parserOptions: {
    ecmaVersion: 5
  },
  extends: [
    './rules/core.js'
  ].map(require.resolve),
  rules: {
    strict: [2, 'function']
  }
};
