module.exports = {
  env: {
    es6: true
  },
  parserOptions: {
    sourceType: 'module',
    ecmaVersion: 6,
    ecmaFeatures: {
      experimentalObjectRestSpread: true // enable support for the experimental object rest/spread properties
    }
  },
  extends: [
    './rules/core.js',
    './rules/es6.js',
  ].map(require.resolve)
};
