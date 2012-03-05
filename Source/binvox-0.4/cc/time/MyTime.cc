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

#include "MyTime.h"





MyTime::MyTime()
{
  set_now();
  
}  // constructor



MyTime::~MyTime()
{

}  // destructor



void
MyTime::set_now()
{
  time(&time_in_seconds);
  calendar_time_ptr = localtime(&time_in_seconds);

}  // MyTime::set_now


  
char *
MyTime::get_time_string()
{
  int nr_chars = strftime(time_string, MAX_TIME_STRING_LENGTH,
			  "%y%m%d_%H%M%S", calendar_time_ptr);

  return time_string;

}  // MyTime::get_time_string



ostream& operator<<(ostream& out, MyTime& T)
{
  out << T.get_time_string();
  return out;

}  // operator<<


