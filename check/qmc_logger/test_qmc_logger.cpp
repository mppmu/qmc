#include "catch.hpp"
#include "qmc.hpp"

#include <sstream> // ostringstream
#include <string> // to_string

TEST_CASE( "Setting Output Stream", "[logger]") {

    SECTION ( "Check Constructor" )
    {
        std::ostringstream stream;
        REQUIRE_NOTHROW( integrators::Logger(stream) );
    };

    SECTION( "Check Assignment" )
    {
        std::ostringstream stream;
        integrators::Logger mylogger(std::cout);
        REQUIRE_NOTHROW( mylogger = stream );
    };

    SECTION( "Check Fields")
    {
        integrators::Logger mylogger(std::cout);
        REQUIRE( mylogger.display_timing == true);
        REQUIRE_NOTHROW( mylogger.display_timing = false );
        REQUIRE( mylogger.display_timing == false);
    };

};

TEST_CASE( "Output", "[logger]") {

    std::ostringstream stream;
    integrators::Logger mylogger(stream);
    mylogger.display_timing = false; // do not print timing information

    SECTION( "String" )
    {
        std::string message("test message string");
        mylogger << message;
        REQUIRE( stream.str() == message );
    }

    SECTION( "Char " )
    {
        mylogger << "c";
        REQUIRE( stream.str() == "c" ); // logger should insert in underlying stream
    };

    SECTION( "Char Array" )
    {
        mylogger << "test message char";
        REQUIRE( stream.str() == "test message char" ); // logger should insert in underlying stream
    };

    SECTION( "basic_ostream" )
    {
        REQUIRE_NOTHROW( mylogger << std::endl );
    };

};
