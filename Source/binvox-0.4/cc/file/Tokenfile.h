//
// $Id: Tokenfile.h,v 1.3 2001/10/21 03:33:06 min Exp min $
//

#ifndef __TOKENFILE_
#define __TOKENFILE_

#include "../math/common.h"
#include "Datafile.h"

static const char OPEN_PAREN = '(';
static const char CLOSE_PAREN = ')';
static const char OPEN_BRACE = '{';
static const char CLOSE_BRACE = '}';
static const char OPEN_BRACKET = '[';
static const char CLOSE_BRACKET = ']';
static const char DOUBLE_QUOTE = '"';
static const char SINGLE_QUOTE = '\'';
static const char BACKSLASH = '\\';
static const char SEMI_COLON = ';';

static const int MAX_TOKEN_LENGTH = 64;

static const char *number_chars = "+-0123456789.abcdefABCDEFeExX";
static char *default_whitespace_chars = ", \t\n\r";

static char default_tokens[][MAX_TOKEN_LENGTH] = { "\"", "'", "#", "[", "]",
						   "\\", "{", "}", ".", ";", "\0" };
static const int MAX_DEFAULT_TOKEN_LENGTH = 2;





class Tokenfile : public Datafile
{

public:

  Tokenfile() : Datafile() {
    standard_tokens = default_tokens;
    max_standard_token_length = MAX_DEFAULT_TOKEN_LENGTH;
    whitespace_chars = default_whitespace_chars;
    looking_ahead = 0;
  }  // constructor

  ~Tokenfile() {}


  void get_token(char *token, int until_whitespace = 0);
  void lookahead_token(char *token);
  int accept_token(const char *wanted);
  int accept_token(const char wanted);
  int get_standard_token(char *token);
  int is_standard_token(int k);

  char *read_until(char until_char);

  void get_number(char *number_string);

  int get_int();
  Float get_float();

  int skip_whitespace();
  int is_whitespace(int k);

  int is_id_char(int k);
  int is_id_firstchar(int k);
  int is_number_char(int k);

  void set_standard_tokens(char (*new_standard_tokens)[MAX_TOKEN_LENGTH]);
  void set_whitespace_chars(char *new_whitespace_chars) {
    whitespace_chars = new_whitespace_chars;
  }

  void skip_comments();

  int get_parse_error() { return parse_error; }


  static bool ignore_cpp_comments;  // if true, ignore everything after //

  
private:

  char (*standard_tokens)[MAX_TOKEN_LENGTH];
  int max_standard_token_length;

  char *whitespace_chars;

  int looking_ahead;
  int parse_error;
  
  char read_until_buffer[128];  // read_until now returns what was read

  
};  // Tokenfile class


#endif

