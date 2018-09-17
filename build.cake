#addin "Cake.Incubator"

#addin "nuget:https://api.nuget.org/v3/index.json?package=Cake.Coveralls"
#tool "nuget:https://api.nuget.org/v3/index.json?package=coveralls.io&version=1.3.4"
#tool "nuget:https://api.nuget.org/v3/index.json?package=OpenCover&version=4.6.519"
#tool "nuget:https://api.nuget.org/v3/index.json?package=ReportGenerator&version=2.4.5"

#tool nuget:?package=vswhere


//////////////////////////////////////////////////////////////////////
// ARGUMENTS
//////////////////////////////////////////////////////////////////////

var target = Argument("target", "Default");
var configuration = Argument("configuration", "Release");
var coverallsToken = Argument("coverallsToken", string.Empty);

//////////////////////////////////////////////////////////////////////
// PREPARATION
//////////////////////////////////////////////////////////////////////

// Define directories.
var buildDir = Directory(configuration);

//////////////////////////////////////////////////////////////////////
// TASKS
//////////////////////////////////////////////////////////////////////

Task("Clean")
    .Does(() =>
{
    CleanDirectories("./**/bin/**");
});

Task("Restore-NuGet-Packages")
    .Does(() =>
{
    DotNetCoreRestore("./NeuralNet.sln");
});

Task("Build")
    .IsDependentOn("Restore-NuGet-Packages")
    .Does(() =>
{
    DotNetCoreBuild("./NeuralNet.sln", new DotNetCoreBuildSettings {
    Verbosity = DotNetCoreVerbosity.Minimal,
    Configuration = configuration
    });

});

Task("Test")
    .IsDependentOn("Build")
    .Does(() =>
{
     var settings = new DotNetCoreTestSettings
     {
         Configuration = configuration,
        ArgumentCustomization = args=>args.Append("/p:CollectCoverage=true /p:CoverletOutputFormat=opencover")
     };
    DotNetCoreTest("NeuralNetLib.Test/NeuralNetLib.Test.csproj", settings);

});

Task("Appveyor")
    .IsDependentOn("Test")
    .Does(() =>
{
    CoverallsIo("NeuralNetLib.Test/coverage.opencover.xml", new CoverallsIoSettings()
    {
        RepoToken = coverallsToken
    });

});

//////////////////////////////////////////////////////////////////////
// TASK TARGETS
//////////////////////////////////////////////////////////////////////

Task("Default")
    .IsDependentOn("Test");

//////////////////////////////////////////////////////////////////////
// EXECUTION
//////////////////////////////////////////////////////////////////////

RunTarget(target);
