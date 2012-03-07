//
// $Id$
//

#include <map>
#include <string.h>
#include <stdlib.h>
#include "ObjMeshFile.h"

static const int MAX_FACES_PER_FILE = 50000;  // for OBJ segment files

#define DEBUG(x)





ObjMeshFile::ObjMeshFile(Mesh& mesh_ref, string filename) :
  MeshFile(mesh_ref, filename)
{

}  // constructor



ObjMeshFile::~ObjMeshFile()
{

}  // destructor



static char obj_tokens[][MAX_TOKEN_LENGTH] = { "-", "\n", "/", "\\" };  // why the \n ?

int
ObjMeshFile::load()
{
  cout << "ObjMeshFile::load(" << my_filename << ")" << endl;

  // keep track of unknown tokens so we don't keep repeating the same error message
  std::map <string, bool> unknown_tokens;

  Tokenfile *in = new Tokenfile();
  if (!in->open(my_filename)) {
    cout << "Error: could not find the file [" << my_filename << "]" << endl;
    return 0;
  }
  Tokenfile::ignore_cpp_comments = false;  // obj file can have // in it..
  char token[MAX_TOKEN_LENGTH];
  mesh.clear();

  in->print_nr_lines();
  int line_nr;

  //
  // trick to make newline character a token
  //
  in->set_whitespace_chars(", \t\r");  // found an obj file with backslashes in it...
  in->set_standard_tokens(obj_tokens);

  Vertex *new_vertex_p;
  Face *new_face_p;

  int group_index = -1;
  
  while(in->get_current_char() != MY_EOF) {
    in->get_token(token);

    if(strcmp(token, "v") == 0) {
      //
      // get (x, y, z) values, add Vertex to vertices array
      //
      float x, y, z;
      in->get_number(token), x = atof(token);
      in->get_number(token), y = atof(token);
      in->get_number(token), z = atof(token);
      new_vertex_p = new Vertex(x, y, z);
      in->accept_token('\n');
      mesh.vertices.push_back(new_vertex_p);
      DEBUG(cout << *new_vertex_p << endl);
    }
    else if (strcmp(token, "f") == 0) {
      new_face_p = new Face();
      
      int done = 0;
      do {
	in->get_token(token);

	if (token[0] == '\\') {  // after a backslash, skip over the newline
	  in->get_token(token);
	  in->get_token(token);
	}

	if (token[0] == '\n') done = 1;
	else {
	  int sign = 1;
	  if (token[0] == '-') {
	    sign = -1;
	    in->get_token(token);
	  }
	  int v_id = sign * atoi(token);

	  //	  cout << "v" << v_id << " ";
	  
	  if (v_id < 0) {
	    new_face_p->add_vertex(mesh.get_nr_vertices() + v_id);
	  }
	  else {
	    new_face_p->add_vertex(v_id - 1);  // .obj indices start at 1 ...
	  }
	  
	  in->lookahead_token(token);
	  
	  if (token[0] == '/') {
	    in->get_token(token);  // get the slash
	    //	    cout << " token1[" << token << "] ";
	    in->lookahead_token(token);  // lookahead because there may be no texture index
	    if (token[0] != '/') {
	      in->get_token(token);  // texture index
	      if (token[0] == '-') in->get_token(token);  // which could also be negative...
	      //	      cout << "t" << token << " ";
	      //
	      // here there could be a normal index
	      //
	    }

	    in->lookahead_token(token);
	    if (token[0] == '/') {
	      in->get_token(token, 1);  // get the slash, until whitespace
	      //	      cout << " token2[" << token << "] ";
	      //	      in->go_back_char();  // why??
	      char k = in->get_buffered_char();  // if it's whitespace then normal index is empty
	      in->go_back_char();  // i.e. lookahead char
	      //	      cout << " buff[" << k << "] ";
	      if (!in->is_whitespace(k)) {
		in->get_token(token);  // normal index, can this be empty too? YES
		if (token[0] == '-') in->get_token(token);  // which could also be negative...
		//		cout << "n" << token << " ";
	      }
	    }
	  }  // if
	
	  //	  cout << endl;
	  
	}  // else, got a vertex
	
      } while (!done);

      mesh.faces.push_back(new_face_p);

      DEBUG(cout << *new_face_p << endl);
      
    }  // face 'f'
    else if (strcmp(token, "g") == 0) {
      group_index++;
      char *rest_of_line = in->read_until('\n');  // found OBJ files with empty group names..
      cout << "  reading group [" << rest_of_line << "]" << endl;
    }
    else if (strcmp(token, "s") == 0) {
      in->get_token(token);
      cout << "  ignoring smooth shading directive [s " << token << "]" << endl;
    }
    else if (strcmp(token, "vt") == 0) {
      // skip texture coordinates (may be 1, 2, or 3 of them)
      in->read_until('\n');
    }
    else if (strcmp(token, "vn") == 0) {
      // vertex normal
      float a, b, c;
      in->get_number(token), a = atof(token);
      in->get_number(token), b = atof(token);
      in->get_number(token), c = atof(token);
    }
    else if (strcmp(token, "mtllib") == 0) {
      // material library
      char filename[128];
      in->read_line(filename);
      //      in->get_token(token);  // get material library filename
      cout << "material library: " << filename << endl;
    }
    else if (strcmp(token, "usemtl") == 0) {
      // use material
      in->get_token(token);  // get material name
      cout << "  ignoring [usemtl " << token << "]" << endl;
    }
    else {
      if (token[0] != '\n') {
	if (!unknown_tokens[token]) {
	  in->print_error_line();
	  cout << "Warning: do not know how to handle [" << token << "]" << endl;
	  in->read_until('\n');  // skip rest of the line
	  unknown_tokens[token] = true;
	}
      }
    }  // unknown token, not a newline

    in->skip_whitespace();  // has to be here in case there's whitespace at the end

    line_nr = in->get_line_nr();
    if ((line_nr % 100) == 0) cout << "\r  " << line_nr << " lines read" << flush;
      
  }  // while not eof

  in->close();

  cout << "\r  " << line_nr << " lines read" << endl;
  cout << "Read " << mesh.faces.size() << " faces, " << mesh.vertices.size() << " vertices." << endl;

  return 1;
	
}  // ObjMeshFile::load



