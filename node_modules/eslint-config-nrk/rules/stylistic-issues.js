module.exports = {
  rules: {
    'array-bracket-spacing': 0,       // http://eslint.org/docs/rules/array-bracket-spacing
    'block-spacing': [2, 'always'],   // http://eslint.org/docs/rules/block-spacing
    'brace-style': 2,                 // http://eslint.org/docs/rules/brace-style
    camelcase: [                    // http://eslint.org/docs/rules/camelcase
      2, {
        properties: 'always'
      }
    ],
    'comma-spacing': [                // http://eslint.org/docs/rules/comma-spacing
      2, {
        before: false,
        after: true
      }
    ],
    'comma-style': [2, 'last'],       // http://eslint.org/docs/rules/comma-style
    'computed-property-spacing': [    // http://eslint.org/docs/rules/computed-property-spacing
      2, 'never'
    ],
    'consistent-this': [2, 'self'],   // http://eslint.org/docs/rules/consistent-this
    'eol-last': 2,                    // http://eslint.org/docs/rules/eol-last
    'func-names': 0,                  // http://eslint.org/docs/rules/func-names
    'func-style': 0,                  // http://eslint.org/docs/rules/func-style
    'id-length': 0,                   // http://eslint.org/docs/rules/id-length,
    'id-match': 0,                    // http://eslint.org/docs/rules/id-match
    indent: [                       // http://eslint.org/docs/rules/indent
      2,
      2,
      {
        SwitchCase: 1
      }
    ],
    'key-spacing': [                  // http://eslint.org/docs/rules/key-spacing
      2, {
        beforeColon: false,
        afterColon: true
      }
    ],
    'keyword-spacing': 0,             // http://eslint.org/docs/rules/keyword-spacing
    'linebreak-style': [2, 'unix'],   // http://eslint.org/docs/rules/linebreak-style
    'lines-around-comment': 2,        // http://eslint.org/docs/rules/lines-around-comment
    'max-depth': [2, 4],              // http://eslint.org/docs/rules/max-depth
    'max-len': [0, 80],               // http://eslint.org/docs/rules/max-len
    'max-nested-callbacks': [2, 3],   // http://eslint.org/docs/rules/max-nested-callbacks
    'max-params': [2, 3],             // http://eslint.org/docs/rules/max-params
    'max-statements': [0, 2],         // http://eslint.org/docs/rules/max-statements
    'max-statements-per-line': [
      2, {
        max: 2
      }
    ],
    'new-cap': 2,                     // http://eslint.org/docs/rules/new-cap
    'new-parens': 2,                  // http://eslint.org/docs/rules/new-parens
    'newline-after-var': 0,           // http://eslint.org/docs/rules/newline-after-var
    'newline-before-return': 0,       // http://eslint.org/docs/rules/newline-before-return
    'no-array-constructor': 2,        // http://eslint.org/docs/rules/no-array-constructor
    'no-bitwise': 2,                  // http://eslint.org/docs/rules/no-bitwise
    'no-continue': 2,                 // http://eslint.org/docs/rules/no-continue
    'no-inline-comments': 0,          // http://eslint.org/docs/rules/no-inline-comments
    'no-lonely-if': 2,                // http://eslint.org/docs/rules/no-lonely-if
    'no-mixed-spaces-and-tabs': [     // http://eslint.org/docs/rules/no-mixed-spaces-and-tabs
      2,
      'smart-tabs'
    ],
    'no-multiple-empty-lines': [      // http://eslint.org/docs/rules/no-multiple-empty-lines
      1,
      {
        max: 3
      }
    ],
    'no-nested-ternary': 2,           // http://eslint.org/docs/rules/no-nested-ternary
    'no-negated-condition': 0,        // http://eslint.org/docs/rules/no-negated-condition
    'no-new-object': 2,               // http://eslint.org/docs/rules/no-new-object
    'no-plusplus': 0,                 // http://eslint.org/docs/rules/no-plusplus
    'no-restricted-syntax': 0,        // http://eslint.org/docs/rules/no-restricted-syntax
    'no-spaced-func': 2,              // http://eslint.org/docs/rules/no-spaced-func
    'no-ternary': 0,                  // http://eslint.org/docs/rules/no-ternary
    'no-trailing-spaces': [           // http://eslint.org/docs/rules/no-trailing-spaces
      2,
      {
        skipBlankLines: true
      }
    ],
    'no-underscore-dangle': 0,        // http://eslint.org/docs/rules/no-underscore-dangle
    'no-unneeded-ternary': 2,         // http://eslint.org/docs/rules/no-unneeded-ternary
    'object-curly-spacing': 0,        // http://eslint.org/docs/rules/object-curly-spacing
    'one-var': 0,                     // http://eslint.org/docs/rules/one-var
    'operator-assignment': 0,         // http://eslint.org/docs/rules/operator-assignment
    'operator-linebreak': [           // http://eslint.org/docs/rules/operator-linebreak
      2,
      'before',
      {
        overrides: {
          '?': 'after'
        }
      }
    ],
    'padded-blocks': 0,               // http://eslint.org/docs/rules/padded-blocks
    'quote-props': [2, 'as-needed'],  // http://eslint.org/docs/rules/quote-props
    quotes: [2, 'single'],          // http://eslint.org/docs/rules/quotes
    'require-jsdoc': 0,               // http://eslint.org/docs/rules/require-jsdoc
    'semi-spacing': 2,                // http://eslint.org/docs/rules/semi-spacing
    semi: [2, 'always'],            // http://eslint.org/docs/rules/semi.html
    'sort-vars': 0,                   // http://eslint.org/docs/rules/sort-vars
    'space-before-blocks': 0,         // http://eslint.org/docs/rules/space-before-blocks
    'space-before-function-paren': 0, // http://eslint.org/docs/rules/space-before-function-paren
    'space-in-parens': 0,             // http://eslint.org/docs/rules/space-in-parens
    'space-infix-ops': 0,             // http://eslint.org/docs/rules/space-infix-ops
    'space-unary-ops': 0,             // http://eslint.org/docs/rules/space-unary-ops
    'spaced-comment': 0,              // http://eslint.org/docs/rules/spaced-comment
    'wrap-regex': 0,                  // http://eslint.org/docs/rules/wrap-regex
  }
};
