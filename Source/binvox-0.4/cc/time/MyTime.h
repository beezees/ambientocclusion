//
// thinvox, a binary voxel thinning program
// Copyright (c) 2004-2012 by Patrick Min, patrick.n.min "at" gmail "dot" com
// 
// This program is free software; you can redistribute it and/or
// modify it under the terms of the GNU General Public License
// as published by the Free Software Foundation; either version 2
// of the License, or (at your option) any later version.
// 
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
// 
// You should have received a copy of the GNU General Public License
// along with this program; if not, write to the Free Software
// Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
// 
//
// $Id$
//

#ifndef __MyTime_h
#define __MyTime_h

#include <iostream>
#include <time.h>

static const int MAX_TIME_STRING_LENGTH = 128;

using namespace std;





class MyTime
{

public:

  MyTime();
  ~MyTime();

  void set_now();
  
  char *get_time_string();

  friend ostream& operator<<(ostream& out, MyTime& T);

  
private:

  time_t time_in_seconds;
  struct tm *calendar_time_ptr;

  char time_string[MAX_TIME_STRING_LENGTH];

  
};  // MyTime class


#endif


