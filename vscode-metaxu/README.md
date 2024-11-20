# Metaxu Language Support for VS Code

This extension provides syntax highlighting for the Metaxu programming language.

## Features

- Syntax highlighting for:
  - Keywords (`fn`, `let`, `struct`, `effect`, etc.)
  - Mode annotations (`@local`, `@mut`, `@const`, etc.)
  - Types (including effects)
  - Functions and effect operations
  - Comments (line and block)
  - Strings and numbers
  - Operators

## Example

```metaxu
effect State<T> {
    fn get() -> T
    fn set(T) -> ()
}

struct Counter {
    @mut value: int,
    @const name: string
}

fn increment(@mut counter: Counter) {
    counter.value = counter.value + 1;
}

fn main() {
    let @local counter = Counter {
        value: 0,
        name: "test"
    };
    
    try {
        perform State::set(42) with |k| {
            k(())
        }
    }
}
```

## Installation

1. Copy the `vscode-metaxu` folder to your VS Code extensions directory:
   - Windows: `%USERPROFILE%\.vscode\extensions`
   - macOS/Linux: `~/.vscode/extensions`

2. Restart VS Code

## License

MIT

## Contributing

Feel free to open issues or submit pull requests on GitHub.
