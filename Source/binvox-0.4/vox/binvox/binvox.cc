//
// binvox, a binary 3D mesh voxelizer
// Copyright (c) 2004-2008 by Patrick Min, patrick.n.min "at" gmail "dot" com
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
// $Id: mc.cc,v 1.13 2002/10/29 20:53:23 min Exp min $
//

#include <iostream>
#include <string>
#ifndef IRIX
#include <cstdlib>
#endif
#include <math.h>
#include <string.h>
#include <assert.h>
#include "cc/geom/MeshFileIdentifier.h"
#include "cc/voxel/Voxelizer.h"
#include "cc/voxel/VoxelFile.h"
#include "cc/time/WallTimer.h"
#include "cc/ui/GLwindow.h"

#include "version.h"

static string model_filespec = "";
static string model_prefix = "";
static string model_extension = "";
static string voxel_extension = "";
static int voxel_type;

static int carving_only = 0;
static int dilated_carving = 0;
static int dilate = 0;
static int voting_only = 0;
static int voxel_dim = Voxelizer::DEFAULT_WIDTH;

static Mesh *mesh_p;

// used to compute reconstruction transform from
// voxel coordinates to mesh coordinates
Vector bmin, bmax;
static int set_bounding_box = 0;
Vector norm_translate;
Float norm_scale;

static GLwindow *ortho_win;

using namespace std;





//
// idle callback for voxelization
//
void voxelize()
{
  //
  // 4th parameter 0 so no types array is allocated in Voxels
  //
  Voxels *voxels = new Voxels(voxel_dim, voxel_dim, voxel_dim, 0);
  if (!voxels) {
    cout << "Error: not enough memory for voxel array" << endl;
    return;
  }
  
  Voxelizer *voxelizer = new Voxelizer(*voxels, *mesh_p, ortho_win);
  voxelizer->set_dilated(dilated_carving);
  
  if (carving_only)
    voxelizer->carve_voxelize();
  else {
    // voting_only = 'skip_simple' parameter, so this function will not call carve_voxelize
    voxelizer->parity_vote_voxelize(voting_only);
  }

  // dilate if requested
  if (dilate) {
    for(int i=0; i < dilate; i++) {
      cout << "  dilation step " << i+1 << endl;
      voxels->dilate();
    }
  }

  // set mesh coord correspondence data
  voxels->set_norm_translate(norm_translate);
  voxels->set_norm_scale(norm_scale);
  
  cout << endl << "writing voxel file..." << endl;
  VoxelFile *voxel_file = new VoxelFile(*voxels, model_prefix);
  voxel_file->open_for_write(voxel_type);
  voxel_file->write_file();
  cout << "done" << endl << endl;
  
  exit(0);
  
}  // voxelize



void print_gpl()
{
  
  cout << "  		    GNU GENERAL PUBLIC LICENSE" << endl;
  cout << "		       Version 2, June 1991" << endl;
  cout << "" << endl;
  cout << " Copyright (C) 1989, 1991 Free Software Foundation, Inc.," << endl;
  cout << " 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA" << endl;
  cout << " Everyone is permitted to copy and distribute verbatim copies" << endl;
  cout << " of this license document, but changing it is not allowed." << endl;
  cout << "" << endl;
  cout << "			    Preamble" << endl;
  cout << "" << endl;
  cout << "  The licenses for most software are designed to take away your" << endl;
  cout << "freedom to share and change it.  By contrast, the GNU General Public" << endl;
  cout << "License is intended to guarantee your freedom to share and change free" << endl;
  cout << "software--to make sure the software is free for all its users.  This" << endl;
  cout << "General Public License applies to most of the Free Software" << endl;
  cout << "Foundation's software and to any other program whose authors commit to" << endl;
  cout << "using it.  (Some other Free Software Foundation software is covered by" << endl;
  cout << "the GNU Lesser General Public License instead.)  You can apply it to" << endl;
  cout << "your programs, too." << endl;
  cout << "" << endl;
  cout << "  When we speak of free software, we are referring to freedom, not" << endl;
  cout << "price.  Our General Public Licenses are designed to make sure that you" << endl;
  cout << "have the freedom to distribute copies of free software (and charge for" << endl;
  cout << "this service if you wish), that you receive source code or can get it" << endl;
  cout << "if you want it, that you can change the software or use pieces of it" << endl;
  cout << "in new free programs; and that you know you can do these things." << endl;
  cout << "" << endl;
  cout << "  To protect your rights, we need to make restrictions that forbid" << endl;
  cout << "anyone to deny you these rights or to ask you to surrender the rights." << endl;
  cout << "These restrictions translate to certain responsibilities for you if you" << endl;
  cout << "distribute copies of the software, or if you modify it." << endl;
  cout << "" << endl;
  cout << "  For example, if you distribute copies of such a program, whether" << endl;
  cout << "gratis or for a fee, you must give the recipients all the rights that" << endl;
  cout << "you have.  You must make sure that they, too, receive or can get the" << endl;
  cout << "source code.  And you must show them these terms so they know their" << endl;
  cout << "rights." << endl;
  cout << "" << endl;
  cout << "  We protect your rights with two steps: (1) copyright the software, and" << endl;
  cout << "(2) offer you this license which gives you legal permission to copy," << endl;
  cout << "distribute and/or modify the software." << endl;
  cout << "" << endl;
  cout << "  Also, for each author's protection and ours, we want to make certain" << endl;
  cout << "that everyone understands that there is no warranty for this free" << endl;
  cout << "software.  If the software is modified by someone else and passed on, we" << endl;
  cout << "want its recipients to know that what they have is not the original, so" << endl;
  cout << "that any problems introduced by others will not reflect on the original" << endl;
  cout << "authors' reputations." << endl;
  cout << "" << endl;
  cout << "  Finally, any free program is threatened constantly by software" << endl;
  cout << "patents.  We wish to avoid the danger that redistributors of a free" << endl;
  cout << "program will individually obtain patent licenses, in effect making the" << endl;
  cout << "program proprietary.  To prevent this, we have made it clear that any" << endl;
  cout << "patent must be licensed for everyone's free use or not licensed at all." << endl;
  cout << "" << endl;
  cout << "  The precise terms and conditions for copying, distribution and" << endl;
  cout << "modification follow." << endl;
  cout << endl;
  cout << "		    GNU GENERAL PUBLIC LICENSE" << endl;
  cout << "   TERMS AND CONDITIONS FOR COPYING, DISTRIBUTION AND MODIFICATION" << endl;
  cout << endl;
  cout << "  0. This License applies to any program or other work which contains" << endl;
  cout << "a notice placed by the copyright holder saying it may be distributed" << endl;
  cout << "under the terms of this General Public License.  The \"Program\", below," << endl;
  cout << "refers to any such program or work, and a \"work based on the Program\"" << endl;
  cout << "means either the Program or any derivative work under copyright law:" << endl;
  cout << "that is to say, a work containing the Program or a portion of it," << endl;
  cout << "either verbatim or with modifications and/or translated into another" << endl;
  cout << "language.  (Hereinafter, translation is included without limitation in" << endl;
  cout << "the term \"modification\".)  Each licensee is addressed as \"you\"." << endl;
  cout << endl;
  cout << "Activities other than copying, distribution and modification are not" << endl;
  cout << "covered by this License; they are outside its scope.  The act of" << endl;
  cout << "running the Program is not restricted, and the output from the Program" << endl;
  cout << "is covered only if its contents constitute a work based on the" << endl;
  cout << "Program (independent of having been made by running the Program)." << endl;
  cout << "Whether that is true depends on what the Program does." << endl;
  cout << endl;
  cout << "  1. You may copy and distribute verbatim copies of the Program's" << endl;
  cout << "source code as you receive it, in any medium, provided that you" << endl;
  cout << "conspicuously and appropriately publish on each copy an appropriate" << endl;
  cout << "copyright notice and disclaimer of warranty; keep intact all the" << endl;
  cout << "notices that refer to this License and to the absence of any warranty;" << endl;
  cout << "and give any other recipients of the Program a copy of this License" << endl;
  cout << "along with the Program." << endl;
  cout << endl;
  cout << "You may charge a fee for the physical act of transferring a copy, and" << endl;
  cout << "you may at your option offer warranty protection in exchange for a fee." << endl;
  cout << endl;
  cout << "  2. You may modify your copy or copies of the Program or any portion" << endl;
  cout << "of it, thus forming a work based on the Program, and copy and" << endl;
  cout << "distribute such modifications or work under the terms of Section 1" << endl;
  cout << "above, provided that you also meet all of these conditions:" << endl;
  cout << endl;
  cout << "    a) You must cause the modified files to carry prominent notices" << endl;
  cout << "    stating that you changed the files and the date of any change." << endl;
  cout << endl;
  cout << "    b) You must cause any work that you distribute or publish, that in" << endl;
  cout << "    whole or in part contains or is derived from the Program or any" << endl;
  cout << "    part thereof, to be licensed as a whole at no charge to all third" << endl;
  cout << "    parties under the terms of this License." << endl;
  cout << endl;
  cout << "    c) If the modified program normally reads commands interactively" << endl;
  cout << "    when run, you must cause it, when started running for such" << endl;
  cout << "    interactive use in the most ordinary way, to print or display an" << endl;
  cout << "    announcement including an appropriate copyright notice and a" << endl;
  cout << "    notice that there is no warranty (or else, saying that you provide" << endl;
  cout << "    a warranty) and that users may redistribute the program under" << endl;
  cout << "    these conditions, and telling the user how to view a copy of this" << endl;
  cout << "    License.  (Exception: if the Program itself is interactive but" << endl;
  cout << "    does not normally print such an announcement, your work based on" << endl;
  cout << "    the Program is not required to print an announcement.)" << endl;
  cout << endl;
  cout << "These requirements apply to the modified work as a whole.  If" << endl;
  cout << "identifiable sections of that work are not derived from the Program," << endl;
  cout << "and can be reasonably considered independent and separate works in" << endl;
  cout << "themselves, then this License, and its terms, do not apply to those" << endl;
  cout << "sections when you distribute them as separate works.  But when you" << endl;
  cout << "distribute the same sections as part of a whole which is a work based" << endl;
  cout << "on the Program, the distribution of the whole must be on the terms of" << endl;
  cout << "this License, whose permissions for other licensees extend to the" << endl;
  cout << "entire whole, and thus to each and every part regardless of who wrote it." << endl;
  cout << endl;
  cout << "Thus, it is not the intent of this section to claim rights or contest" << endl;
  cout << "your rights to work written entirely by you; rather, the intent is to" << endl;
  cout << "exercise the right to control the distribution of derivative or" << endl;
  cout << "collective works based on the Program." << endl;
  cout << endl;
  cout << "In addition, mere aggregation of another work not based on the Program" << endl;
  cout << "with the Program (or with a work based on the Program) on a volume of" << endl;
  cout << "a storage or distribution medium does not bring the other work under" << endl;
  cout << "the scope of this License." << endl;
  cout << endl;
  cout << "  3. You may copy and distribute the Program (or a work based on it," << endl;
  cout << "under Section 2) in object code or executable form under the terms of" << endl;
  cout << "Sections 1 and 2 above provided that you also do one of the following:" << endl;
  cout << endl;
  cout << "    a) Accompany it with the complete corresponding machine-readable" << endl;
  cout << "    source code, which must be distributed under the terms of Sections" << endl;
  cout << "    1 and 2 above on a medium customarily used for software interchange; or," << endl;
  cout << endl;
  cout << "    b) Accompany it with a written offer, valid for at least three" << endl;
  cout << "    years, to give any third party, for a charge no more than your" << endl;
  cout << "    cost of physically performing source distribution, a complete" << endl;
  cout << "    machine-readable copy of the corresponding source code, to be" << endl;
  cout << "    distributed under the terms of Sections 1 and 2 above on a medium" << endl;
  cout << "    customarily used for software interchange; or," << endl;
  cout << endl;
  cout << "    c) Accompany it with the information you received as to the offer" << endl;
  cout << "    to distribute corresponding source code.  (This alternative is" << endl;
  cout << "    allowed only for noncommercial distribution and only if you" << endl;
  cout << "    received the program in object code or executable form with such" << endl;
  cout << "    an offer, in accord with Subsection b above.)" << endl;
  cout << endl;
  cout << "The source code for a work means the preferred form of the work for" << endl;
  cout << "making modifications to it.  For an executable work, complete source" << endl;
  cout << "code means all the source code for all modules it contains, plus any" << endl;
  cout << "associated interface definition files, plus the scripts used to" << endl;
  cout << "control compilation and installation of the executable.  However, as a" << endl;
  cout << "special exception, the source code distributed need not include" << endl;
  cout << "anything that is normally distributed (in either source or binary" << endl;
  cout << "form) with the major components (compiler, kernel, and so on) of the" << endl;
  cout << "operating system on which the executable runs, unless that component" << endl;
  cout << "itself accompanies the executable." << endl;
  cout << endl;
  cout << "If distribution of executable or object code is made by offering" << endl;
  cout << "access to copy from a designated place, then offering equivalent" << endl;
  cout << "access to copy the source code from the same place counts as" << endl;
  cout << "distribution of the source code, even though third parties are not" << endl;
  cout << "compelled to copy the source along with the object code." << endl;
  cout << endl;
  cout << "  4. You may not copy, modify, sublicense, or distribute the Program" << endl;
  cout << "except as expressly provided under this License.  Any attempt" << endl;
  cout << "otherwise to copy, modify, sublicense or distribute the Program is" << endl;
  cout << "void, and will automatically terminate your rights under this License." << endl;
  cout << "However, parties who have received copies, or rights, from you under" << endl;
  cout << "this License will not have their licenses terminated so long as such" << endl;
  cout << "parties remain in full compliance." << endl;
  cout << endl;
  cout << "  5. You are not required to accept this License, since you have not" << endl;
  cout << "signed it.  However, nothing else grants you permission to modify or" << endl;
  cout << "distribute the Program or its derivative works.  These actions are" << endl;
  cout << "prohibited by law if you do not accept this License.  Therefore, by" << endl;
  cout << "modifying or distributing the Program (or any work based on the" << endl;
  cout << "Program), you indicate your acceptance of this License to do so, and" << endl;
  cout << "all its terms and conditions for copying, distributing or modifying" << endl;
  cout << "the Program or works based on it." << endl;
  cout << endl;
  cout << "  6. Each time you redistribute the Program (or any work based on the" << endl;
  cout << "Program), the recipient automatically receives a license from the" << endl;
  cout << "original licensor to copy, distribute or modify the Program subject to" << endl;
  cout << "these terms and conditions.  You may not impose any further" << endl;
  cout << "restrictions on the recipients' exercise of the rights granted herein." << endl;
  cout << "You are not responsible for enforcing compliance by third parties to" << endl;
  cout << "this License." << endl;
  cout << endl;
  cout << "  7. If, as a consequence of a court judgment or allegation of patent" << endl;
  cout << "infringement or for any other reason (not limited to patent issues)," << endl;
  cout << "conditions are imposed on you (whether by court order, agreement or" << endl;
  cout << "otherwise) that contradict the conditions of this License, they do not" << endl;
  cout << "excuse you from the conditions of this License.  If you cannot" << endl;
  cout << "distribute so as to satisfy simultaneously your obligations under this" << endl;
  cout << "License and any other pertinent obligations, then as a consequence you" << endl;
  cout << "may not distribute the Program at all.  For example, if a patent" << endl;
  cout << "license would not permit royalty-free redistribution of the Program by" << endl;
  cout << "all those who receive copies directly or indirectly through you, then" << endl;
  cout << "the only way you could satisfy both it and this License would be to" << endl;
  cout << "refrain entirely from distribution of the Program." << endl;
  cout << endl;
  cout << "If any portion of this section is held invalid or unenforceable under" << endl;
  cout << "any particular circumstance, the balance of the section is intended to" << endl;
  cout << "apply and the section as a whole is intended to apply in other" << endl;
  cout << "circumstances." << endl;
  cout << endl;
  cout << "It is not the purpose of this section to induce you to infringe any" << endl;
  cout << "patents or other property right claims or to contest validity of any" << endl;
  cout << "such claims; this section has the sole purpose of protecting the" << endl;
  cout << "integrity of the free software distribution system, which is" << endl;
  cout << "implemented by public license practices.  Many people have made" << endl;
  cout << "generous contributions to the wide range of software distributed" << endl;
  cout << "through that system in reliance on consistent application of that" << endl;
  cout << "system; it is up to the author/donor to decide if he or she is willing" << endl;
  cout << "to distribute software through any other system and a licensee cannot" << endl;
  cout << "impose that choice." << endl;
  cout << endl;
  cout << "This section is intended to make thoroughly clear what is believed to" << endl;
  cout << "be a consequence of the rest of this License." << endl;
  cout << endl;
  cout << "  8. If the distribution and/or use of the Program is restricted in" << endl;
  cout << "certain countries either by patents or by copyrighted interfaces, the" << endl;
  cout << "original copyright holder who places the Program under this License" << endl;
  cout << "may add an explicit geographical distribution limitation excluding" << endl;
  cout << "those countries, so that distribution is permitted only in or among" << endl;
  cout << "countries not thus excluded.  In such case, this License incorporates" << endl;
  cout << "the limitation as if written in the body of this License." << endl;
  cout << endl;
  cout << "  9. The Free Software Foundation may publish revised and/or new versions" << endl;
  cout << "of the General Public License from time to time.  Such new versions will" << endl;
  cout << "be similar in spirit to the present version, but may differ in detail to" << endl;
  cout << "address new problems or concerns." << endl;
  cout << endl;
  cout << "Each version is given a distinguishing version number.  If the Program" << endl;
  cout << "specifies a version number of this License which applies to it and \"any" << endl;
  cout << "later version\", you have the option of following the terms and conditions" << endl;
  cout << "either of that version or of any later version published by the Free" << endl;
  cout << "Software Foundation.  If the Program does not specify a version number of" << endl;
  cout << "this License, you may choose any version ever published by the Free Software" << endl;
  cout << "Foundation." << endl;
  cout << endl;
  cout << "  10. If you wish to incorporate parts of the Program into other free" << endl;
  cout << "programs whose distribution conditions are different, write to the author" << endl;
  cout << "to ask for permission.  For software which is copyrighted by the Free" << endl;
  cout << "Software Foundation, write to the Free Software Foundation; we sometimes" << endl;
  cout << "make exceptions for this.  Our decision will be guided by the two goals" << endl;
  cout << "of preserving the free status of all derivatives of our free software and" << endl;
  cout << "of promoting the sharing and reuse of software generally." << endl;
  cout << endl;
  cout << "			    NO WARRANTY" << endl;
  cout << endl;
  cout << "  11. BECAUSE THE PROGRAM IS LICENSED FREE OF CHARGE, THERE IS NO WARRANTY" << endl;
  cout << "FOR THE PROGRAM, TO THE EXTENT PERMITTED BY APPLICABLE LAW.  EXCEPT WHEN" << endl;
  cout << "OTHERWISE STATED IN WRITING THE COPYRIGHT HOLDERS AND/OR OTHER PARTIES" << endl;
  cout << "PROVIDE THE PROGRAM \"AS IS\" WITHOUT WARRANTY OF ANY KIND, EITHER EXPRESSED" << endl;
  cout << "OR IMPLIED, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF" << endl;
  cout << "MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE.  THE ENTIRE RISK AS" << endl;
  cout << "TO THE QUALITY AND PERFORMANCE OF THE PROGRAM IS WITH YOU.  SHOULD THE" << endl;
  cout << "PROGRAM PROVE DEFECTIVE, YOU ASSUME THE COST OF ALL NECESSARY SERVICING," << endl;
  cout << "REPAIR OR CORRECTION." << endl;
  cout << endl;
  cout << "  12. IN NO EVENT UNLESS REQUIRED BY APPLICABLE LAW OR AGREED TO IN WRITING" << endl;
  cout << "WILL ANY COPYRIGHT HOLDER, OR ANY OTHER PARTY WHO MAY MODIFY AND/OR" << endl;
  cout << "REDISTRIBUTE THE PROGRAM AS PERMITTED ABOVE, BE LIABLE TO YOU FOR DAMAGES," << endl;
  cout << "INCLUDING ANY GENERAL, SPECIAL, INCIDENTAL OR CONSEQUENTIAL DAMAGES ARISING" << endl;
  cout << "OUT OF THE USE OR INABILITY TO USE THE PROGRAM (INCLUDING BUT NOT LIMITED" << endl;
  cout << "TO LOSS OF DATA OR DATA BEING RENDERED INACCURATE OR LOSSES SUSTAINED BY" << endl;
  cout << "YOU OR THIRD PARTIES OR A FAILURE OF THE PROGRAM TO OPERATE WITH ANY OTHER" << endl;
  cout << "PROGRAMS), EVEN IF SUCH HOLDER OR OTHER PARTY HAS BEEN ADVISED OF THE" << endl;
  cout << "POSSIBILITY OF SUCH DAMAGES." << endl;
  cout << endl;
  cout << "		     END OF TERMS AND CONDITIONS" << endl;
  cout << endl;
  cout << "         How to Apply These Terms to Your New Programs" << endl;
  cout << endl;
  cout << "  If you develop a new program, and you want it to be of the greatest" << endl;
  cout << "possible use to the public, the best way to achieve this is to make it" << endl;
  cout << "free software which everyone can redistribute and change under these terms." << endl;
  cout << endl;
  cout << "  To do so, attach the following notices to the program.  It is safest" << endl;
  cout << "to attach them to the start of each source file to most effectively" << endl;
  cout << "convey the exclusion of warranty; and each file should have at least" << endl;
  cout << "the \"copyright\" line and a pointer to where the full notice is found." << endl;
  cout << endl;
  cout << "    <one line to give the program's name and a brief idea of what it does.>" << endl;
  cout << "    Copyright (C) <year>  <name of author>" << endl;
  cout << endl;
  cout << "    This program is free software; you can redistribute it and/or modify" << endl;
  cout << "    it under the terms of the GNU General Public License as published by" << endl;
  cout << "    the Free Software Foundation; either version 2 of the License, or" << endl;
  cout << "    (at your option) any later version." << endl;
  cout << endl;
  cout << "    This program is distributed in the hope that it will be useful," << endl;
  cout << "    but WITHOUT ANY WARRANTY; without even the implied warranty of" << endl;
  cout << "    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the" << endl;
  cout << "    GNU General Public License for more details." << endl;
  cout << endl;
  cout << "    You should have received a copy of the GNU General Public License along" << endl;
  cout << "    with this program; if not, write to the Free Software Foundation, Inc.," << endl;
  cout << "    51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA." << endl;
  cout << endl;
  cout << "Also add information on how to contact you by electronic and paper mail." << endl;
  cout << endl;
  cout << "If the program is interactive, make it output a short notice like this" << endl;
  cout << "when it starts in an interactive mode:" << endl;
  cout << endl;
  cout << "    Gnomovision version 69, Copyright (C) year name of author" << endl;
  cout << "    Gnomovision comes with ABSOLUTELY NO WARRANTY; for details type `show w'." << endl;
  cout << "    This is free software, and you are welcome to redistribute it" << endl;
  cout << "    under certain conditions; type `show c' for details." << endl;
  cout << endl;
  cout << "The hypothetical commands `show w' and `show c' should show the appropriate" << endl;
  cout << "parts of the General Public License.  Of course, the commands you use may" << endl;
  cout << "be called something other than `show w' and `show c'; they could even be" << endl;
  cout << "mouse-clicks or menu items--whatever suits your program." << endl;
  cout << endl;
  cout << "You should also get your employer (if you work as a programmer) or your" << endl;
  cout << "school, if any, to sign a \"copyright disclaimer\" for the program, if" << endl;
  cout << "necessary.  Here is a sample; alter the names:" << endl;
  cout << endl;
  cout << "  Yoyodyne, Inc., hereby disclaims all copyright interest in the program" << endl;
  cout << "  `Gnomovision' (which makes passes at compilers) written by James Hacker." << endl;
  cout << endl;
  cout << "  <signature of Ty Coon>, 1 April 1989" << endl;
  cout << "  Ty Coon, President of Vice" << endl;
  cout << endl;
  cout << "This General Public License does not permit incorporating your program into" << endl;
  cout << "proprietary programs.  If your program is a subroutine library, you may" << endl;
  cout << "consider it more useful to permit linking proprietary applications with the" << endl;
  cout << "library.  If this is what you want to do, use the GNU Lesser General" << endl;
  cout << "Public License instead of this License." << endl;
  cout << endl;
  
}  // print_gpl



int parse_args(int argc, char **argv)
{
  argc--;
  argv++;

  while (argc > 0) {
    if ((*argv)[0] == '-') {
      if (!strcmp(*argv, "-t")) {
	argv++; argc--;
	voxel_extension = *argv;
	voxel_type = VoxelFile::get_filetype(voxel_extension);
	if (voxel_type == -1) {
	  cout << "Error: unknown voxel file type [" << voxel_extension << "]" << endl;
	  return 0;
	}
      }
      else if (!strcmp(*argv, "-v")) {
	voting_only = 1;
      }
      else if (!strcmp(*argv, "-c")) {
	carving_only = 1;
      }
      else if (!strcmp(*argv, "-dc")) {
	dilated_carving = 1;
      }
      else if (!strcmp(*argv, "-dilate")) {
	argv++, argc--;
	dilate = atoi(*argv);
	cout << "  will dilate voxel model " << dilate << " times" << endl;
      }
      else if (!strcmp(*argv, "-bb")) {
	argv++, argc--;
	bmin[X] = atof(*argv);
	argv++, argc--;
	bmin[Y] = atof(*argv);
	argv++, argc--;
	bmin[Z] = atof(*argv);
	argv++, argc--;
	bmax[X] = atof(*argv);
	argv++, argc--;
	bmax[Y] = atof(*argv);
	argv++, argc--;
	bmax[Z] = atof(*argv);
	cout << "  will force bounding box to be " << bmin << " - " << bmax << endl;
	set_bounding_box = 1;
      }
      else if (!strcmp(*argv, "-d")) {
	argv++; argc--;
	voxel_dim = atoi(*argv);
      }
      else if (!strcmp(*argv, "-gpl")) {
	print_gpl();
	exit(0);
      }
      else {
	cout << "Error: invalid program argument: [" << *argv << "]" << endl;
	return 0;
      }
    }
    else {
      model_filespec = *argv;
    }
    argv++; argc--;
  }  // while

  if (!model_filespec.length()) {
    cout << "Error: missing model filename" << endl;
    return 0;
  }
  if (voting_only && carving_only) {
    cout << "you specified both carving and voting method (which is the default)" << endl;
  }
  return 1;

}  // parse_args



void split_model_filespec()
{
  model_prefix = model_filespec;
  int dot_pos = model_prefix.rfind('.');
  if (dot_pos != string::npos) {
    model_extension = model_prefix.substr(dot_pos + 1);
    model_prefix.erase(dot_pos);
  }
  else
    model_extension = "";
  
}  // split_model_filespec



int load_model_file()
{
  split_model_filespec();

  if (model_extension.length() == 0) {
    cout << "load_model_file error: missing extension in ["
	 <<  model_filespec << "]" << endl;
    return 0;
  }

  mesh_p = new Mesh();
  mesh_p->clear();
  mesh_p->init();
  mesh_p->set_name(model_prefix);
  
  MeshFileIdentifier meshfile_identifier(*mesh_p);
  MeshFile *meshfile_p = meshfile_identifier.create_mesh_file(model_filespec);
  int result = meshfile_p->load();
  if (!result) return 0;
  
  if (mesh_p->get_nr_faces() == 0) {
    cout << "Error: mesh has no polygons" << endl;
    return 0;
  }

  if (mesh_p) {
    if (mesh_p->compute_bounding_box()) {

      if (set_bounding_box) {
	cout << "  setting bounding box to " << bmin << " - " << bmax << endl;
	mesh_p->set_bounding_box(bmin, bmax);
      }

      norm_scale = 1.0 / mesh_p->normalize();

      mesh_p->get_bounding_box(&bmin, &bmax);

      norm_translate[X] = -0.025 * norm_scale + bmin[X];
      norm_translate[Y] = -0.025 * norm_scale + bmin[Y];
      norm_translate[Z] = -0.025 * norm_scale + bmin[Z];
      
    }
    else {
      cout << "Error: could not compute bounding box (coordinates out of range)" << endl;
      return 0;  // no vertices to compute bbox from
    }
    
  }  // if a mesh was loaded

  return result;

}  // load_model_file





int main(int argc, char **argv)
{
  cout << endl;
  cout << "--- [binvox] mesh voxelizer, version " << binvox_version << ", build #" << binvox_build_number << " on " << binvox_date << endl;
  cout << "--- written by Patrick Min, 2004-2008" << endl;
  cout << "--- binvox comes with ABSOLUTELY NO WARRANTY; for details run 'binvox -gpl'" << endl;
  cout << "--- binvox is free software, and you are welcome to redistribute it under certain conditions; for details run 'binvox -gpl'"
       << endl;
  cout << endl;

  int args_ok = parse_args(argc, argv);

  if (!args_ok) {
    cout << endl;
    cout << "Usage: binvox [-d <voxel dimension>] [-t <voxel file type>] [-c] [-v] <model filespec>" << endl;
    cout << "  -d: specify voxel grid size (default " << Voxelizer::DEFAULT_WIDTH
	 << ", max " << Voxelizer::MAX_WIDTH << ")" << endl;
    cout << "  -t: specify voxel file type (default binvox, also supported: mira, vtk, raw)" << endl;
    cout << "  -c: z-buffer based carving method only" << endl;
    cout << "  -v: z-buffer based parity voting method only (default is both -c and -v)" << endl;
    cout << "Additional parameters:" << endl;
    cout << "  -bb <minx> <miny> <minz> <maxx> <maxy> <maxz>: force a different bounding box" << endl;
    cout << "Supported 3D model file formats:" << endl;
    cout << "  OBJ" << endl;
    cout << "Example:" << endl;
    cout << "binvox -c -d 200 -t mira plane.obj" << endl;
    cout << endl;
  }

  if (!args_ok) return 1;
  
  if (voxel_dim > Voxelizer::MAX_WIDTH) {
    cout << "Error: max voxel grid size is " << Voxelizer::MAX_WIDTH << endl << endl;
    return 1;
  }
  
  glutInit(&argc, argv);

  //
  // load model file
  //
  cout << "loading model file..." << endl;
  if (!load_model_file()) {
    cout << endl;
    return 1;
  }

  //
  // voxelize
  //
  cout << endl << "voxel model dimension: " << voxel_dim << endl;
  cout << "CC: good ..." << endl;
  ortho_win = new GLwindow("binvox", 0, 0, voxel_dim, voxel_dim);
  cout << "CC: set window" << endl;
  glutIdleFunc(voxelize);
  
  //
  // start glut main loop, so voxelize func can be called
  //
  glutMainLoop();
  
  return 0;
  
}  // main

