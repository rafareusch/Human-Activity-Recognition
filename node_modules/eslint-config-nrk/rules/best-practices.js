module.exports = {
  rules: {
    'accessor-pairs': 2,              // http://eslint.org/docs/rules/accessor-pairs
    'block-scoped-var': 2,            // http://eslint.org/docs/rules/block-scoped-var
    complexity: [2, 9],             // http://eslint.org/docs/rules/complexity
    'consistent-return': 2,           // http://eslint.org/docs/rules/consistent-return
    curly: 2,                       // http://eslint.org/docs/rules/curly
    'default-case': 2,                // http://eslint.org/docs/rules/default-case
    'dot-notation': [                 // http://eslint.org/docs/rules/dot-notation
      2, {
        allowKeywords: true
      }
    ],
    'dot-location': 0,                // http://eslint.org/docs/rules/dot-location
    eqeqeq: 2,                      // http://eslint.org/docs/rules/eqeqeq
    'guard-for-in': 2,                // http://eslint.org/docs/rules/guard-for-in
    'no-alert': 2,                    // http://eslint.org/docs/rules/no-alert
    'no-caller': 2,                   // http://eslint.org/docs/rules/no-caller
    'no-case-declarations': 2,        // http://eslint.org/docs/rules/no-case-declarations
    'no-div-regex': 2,                // http://eslint.org/docs/rules/no-div-regex

    'no-empty-pattern': 2,            // http://eslint.org/docs/rules/no-empty-pattern
    'no-else-return': 2,              // http://eslint.org/docs/rules/no-else-return
    'no-eq-null': 2,                  // http://eslint.org/docs/rules/no-eq-null
    'no-eval': 2,                     // http://eslint.org/docs/rules/no-eval
    'no-extend-native': 2,            // http://eslint.org/docs/rules/no-extend-native
    'no-extra-bind': 2,               // http://eslint.org/docs/rules/no-extra-bind
    'no-fallthrough': 2,              // http://eslint.org/docs/rules/no-fallthrough
    'no-floating-decimal': 2,         // http://eslint.org/docs/rules/no-floating-decimal
    'no-invalid-this': 2,             // http://eslint.org/docs/rules/no-invalid-this
    'no-implicit-coercion': [         // http://eslint.org/docs/rules/no-implicit-coercion
      2, {
        boolean: false,
        number: true,
        string: true
      }
    ],
    'no-implied-eval': 2,             // http://eslint.org/docs/rules/no-implied-eval
    'no-iterator': 2,                 // http://eslint.org/docs/rules/no-iterator
    'no-labels': 2,                   // http://eslint.org/docs/rules/no-labels
    'no-lone-blocks': 2,              // http://eslint.org/docs/rules/no-lone-blocks
    'no-loop-func': 2,                // http://eslint.org/docs/rules/no-loop-func
    'no-magic-numbers': 0,            // http://eslint.org/docs/rules/no-magic-numbers
    'no-multi-spaces': 2,             // http://eslint.org/docs/rules/no-multi-spaces
    'no-multi-str': 2,                // http://eslint.org/docs/rules/no-multi-str
    'no-native-reassign': 2,          // http://eslint.org/docs/rules/no-native-reassign
    'no-new': 2,                      // http://eslint.org/docs/rules/no-new
    'no-new-func': 2,                 // http://eslint.org/docs/rules/no-new-func
    'no-new-wrappers': 2,             // http://eslint.org/docs/rules/no-new-wrappers
    'no-octal': 2,                    // http://eslint.org/docs/rules/no-octal
    'no-octal-escape': 2,             // http://eslint.org/docs/rules/no-octal-escape
    'no-param-reassign': 1,           // http://eslint.org/docs/rules/no-param-reassign
    'no-process-env': 1,              // http://eslint.org/docs/rules/no-process-env
    'no-proto': 2,                    // http://eslint.org/docs/rules/no-proto
    'no-redeclare': 2,                // http://eslint.org/docs/rules/no-redeclare
    'no-return-assign': 2,            // http://eslint.org/docs/rules/no-return-assign
    'no-script-url': 2,               // http://eslint.org/docs/rules/no-script-url
    'no-self-assign': 2,              // http://eslint.org/docs/rules/no-self-assign
    'no-self-compare': 2,             // http://eslint.org/docs/rules/no-self-compare
    'no-sequences': 2,                // http://eslint.org/docs/rules/no-sequences
    'no-throw-literal': 2,            // http://eslint.org/docs/rules/no-throw-literal
    'no-unused-expressions': [        // http://eslint.org/docs/rules/no-unused-expressions
      2, {
        allowShortCircuit: true,
        allowTernary: true
      }
    ],
    'no-unused-labels': 2,            // http://eslint.org/docs/rules/no-unused-labels
    'no-useless-call': 2,             // http://eslint.org/docs/rules/no-useless-call
    'no-useless-concat': 2,           // http://eslint.org/docs/rules/no-useless-concat
    'no-useless-escape': 2,           // http://eslint.org/docs/rules/no-useless-escape
    'no-void': 2,                     // http://eslint.org/docs/rules/no-void
    'no-warning-comments': 2,         // http://eslint.org/docs/rules/no-warning-comments
    'no-with': 2,                     // http://eslint.org/docs/rules/no-with
    radix: 2,                         // http://eslint.org/docs/rules/radix
    'vars-on-top': 0,                 // http://eslint.org/docs/rules/vars-on-top
    'wrap-iife': [2, 'any'],          // http://eslint.org/docs/rules/wrap-iife
    yoda: 2,                          // http://eslint.org/docs/rules/yoda
  }
};
