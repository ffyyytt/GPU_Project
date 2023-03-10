#ifndef __IMAGE_HPP__
#define __IMAGE_HPP__

#include <string>
#include <cstring>
#include <stdexcept>

class Image
{
  public:
	Image() = default;
    Image( const int p_width, const int p_height, const int p_nbChannels );
	~Image();

	void load( const std::string & p_path );
	void save( const std::string & p_path ) const;

  public: // All public!
	int				_width		= 0;
	int				_height		= 0;
	int				_nbChannels = 0;
	unsigned char * _pixels		= nullptr;
};

#endif // __IMAGE_HPP__
