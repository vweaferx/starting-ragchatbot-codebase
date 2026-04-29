export default [
    {
        files: ['*.js'],
        languageOptions: {
            ecmaVersion: 2022,
            sourceType: 'script',
            globals: {
                document: 'readonly',
                window: 'readonly',
                console: 'readonly',
                fetch: 'readonly',
                marked: 'readonly',
                Date: 'readonly',
            },
        },
        rules: {
            'no-var': 'error',
            'prefer-const': 'error',
            'no-unused-vars': ['warn', { argsIgnorePattern: '^_' }],
            'no-console': ['warn', { allow: ['error', 'warn'] }],
            eqeqeq: ['error', 'always'],
            curly: ['error', 'all'],
            'no-implicit-globals': 'error',
        },
    },
];
