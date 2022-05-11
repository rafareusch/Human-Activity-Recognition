module.exports = {
  rules: {
    'init-declarations': 0,           // http://eslint.org/docs/rules/init-declarations
    'no-catch-shadow': 2,             // http://eslint.org/docs/rules/no-catch-shadow
    'no-delete-var': 2,               // http://eslint.org/docs/rules/no-delete-var
    'no-label-var': 2,                // http://eslint.org/docs/rules/no-label-var
    'no-shadow-restricted-names': 2,  // http://eslint.org/docs/rules/no-shadow-restricted-names
    'no-shadow': 0,                   // http://eslint.org/docs/rules/no-shadow
    'no-undef-init': 2,               // http://eslint.org/docs/rules/no-undef-init
    'no-undef': 2,                    // http://eslint.org/docs/rules/no-undef
    'no-undefined': 2,                // http://eslint.org/docs/rules/no-undefined
    'no-unused-vars': [2, {           // http://eslint.org/docs/rules/no-unused-vars
      vars: 'all',                    // ... checks all variables including those in global scope
      args: 'after-used'              // ... only the last argument must be used
    }],
    'no-use-before-define': [2, 'nofunc'], // http://eslint.org/docs/rules/no-use-before-define
  }
};
