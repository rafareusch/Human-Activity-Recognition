module.exports = {
  parserOptions: {
    ecmaFeatures: {
      jsx: true
    }
  },
  rules: {
    'react/jsx-boolean-value': 0,                          // github.com/yannickcr/eslint-plugin-react/blob/master/docs/rules/jsx-boolean-value.md
    'react/jsx-closing-bracket-location': 0,               // github.com/yannickcr/eslint-plugin-react/blob/master/docs/rules/jsx-closing-bracket-location.md
    'react/jsx-curly-spacing': [2, 'never'],               // github.com/yannickcr/eslint-plugin-react/blob/master/docs/rules/jsx-curly-spacing.md
    'react/jsx-equals-spacing': [2, 'never'],              // github.com/yannickcr/eslint-plugin-react/blob/master/docs/rules/jsx-equals-spacing.md
    'jsx-first-prop-new-line': 0,                          // github.com/yannickcr/eslint-plugin-react/blob/master/docs/rules/jsx-first-prop-new-line.md
    'react/jsx-handler-names': 1,                          // github.com/yannickcr/eslint-plugin-react/blob/master/docs/rules/jsx-handler-names.md
    'react/jsx-indent': [2, 2],                            // github.com/yannickcr/eslint-plugin-react/blob/master/docs/rules/jsx-indent.md
    'react/jsx-indent-props': [                            // github.com/yannickcr/eslint-plugin-react/blob/master/docs/rules/jsx-indent-props.md
      0,
      0
    ],
    'react/jsx-key': 2,                                    // github.com/yannickcr/eslint-plugin-react/blob/master/docs/rules/jsx-key.md
    'react/jsx-max-props-per-line': [                      // github.com/yannickcr/eslint-plugin-react/blob/master/docs/rules/jsx-max-props-per-line.md
      2, {
        maximum: 6
      }
    ],
    'react/jsx-no-duplicate-props': 2,                     // github.com/yannickcr/eslint-plugin-react/blob/master/docs/rules/jsx-no-duplicate-props.md
    'react/jsx-no-bind': 2,                                // github.com/yannickcr/eslint-plugin-react/blob/master/docs/rules/jsx-no-bind.md
    'react/jsx-no-literals': 0,                            // github.com/yannickcr/eslint-plugin-react/blob/master/docs/rules/jsx-no-literals.md
    'react/jsx-no-undef': 2,                               // github.com/yannickcr/eslint-plugin-react/blob/master/docs/rules/jsx-no-undef.md
    'jsx-quotes': [2, 'prefer-double'],                    // http://eslint.org/docs/rules/jsx-quotes
    'react/jsx-pascal-case': 2,                            // github.com/yannickcr/eslint-plugin-react/blob/master/docs/rules/jsx-pascal-case.md
    'react/jsx-sort-props': 0,                             // github.com/yannickcr/eslint-plugin-react/blob/master/docs/rules/jsx-sort-props.md
    'react/jsx-space-before-closing': [2, 'always'],       // github.com/yannickcr/eslint-plugin-react/blob/master/docs/rules/jsx-space-before-closing.md
    'react/jsx-uses-react': 2,                             // github.com/yannickcr/eslint-plugin-react/blob/master/docs/rules/jsx-uses-react.md
    'react/jsx-uses-vars': 2                               // github.com/yannickcr/eslint-plugin-react/blob/master/docs/rules/jsx-uses-vars.md
  }

};
