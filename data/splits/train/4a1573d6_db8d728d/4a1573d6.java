import java.util.*;
import java.math.*;
import java.io.*; 
 public class A{

static FastReader scan=new FastReader();
 public static PrintWriter out = new PrintWriter (new BufferedOutputStream(System.out));
  static LinkedList<Integer>edges[];
  static boolean stdin = true;
    static String filein = "input";
    static String fileout = "output";
static int dx[] = { -1, 0, 1, 0 };
static int dy[] = { 0, 1, 0, -1 };
int dx_8[]={1,1,1,0,0,-1,-1,-1};
int dy_8[]={-1,0,1,-1,1,-1,0,1};
static char sts[]={'U','R','D','L'};
static boolean prime[];

    
   
 
   
  
 
 
           
           static int perm=0,FOR=0;
           
     

static boolean flag=false;
static int len=100000000;

static ArrayList<Pair>inters=new ArrayList<Pair>();





static StringBuilder sb;



 private static int next(int[] arr, int target)  
    {  
        int start = 0, end = arr.length - 1;  
    
        int ans = -1;  
        while (start <= end) {  
            int mid = (start + end) / 2;  
    
            // Move to right side if target is  
            // greater.  
            if(arr[mid]==target)
                return mid;
            if (arr[mid] <target) {  
                start = mid + 1;  
            }  
    
            // Move left side.  
            else {  
                ans = mid;  
                end = mid - 1;  
            }  
        }  
        return ans;  
    }  
//static boolean vis[][];
    

  

  
  static StringBuilder a,b;

    
static ArrayList<Integer>v;
static ArrayList<Integer>pows;
 


   static long mod=(long)(1e9)+7;
    
        
    static int dp[][];
    static int n;
    static int arr[];
        
        
public static void main(String[] args)  throws Exception 
{
//SUCK IT UP AND DO IT ALRIGHT
//scan=new FastReader("div7.in");
//out = new PrintWriter("div7.out");
   //System.out.println(countSetBits(2015));
    //int elem[]={1,2,3,4,5};
//System.out.println("avjsmlfpb".compareTo("avjsmbpfl"));
int tt=1;

/*for(int i=0;i<=100;i++)
    if(prime[i])
        arr.add(i);
    System.out.println(arr.size());*/
   // check(new StringBuilder("05:11"));
   // System.out.println(26010000000000L%150);
   tt=scan.nextInt();

outer:while(tt-->0)
{
 int n=scan.nextInt(),m=scan.nextInt();
 char board[][]=new char[n][m];
 for(int i=0;i<n;i++)
    board[i]=scan.next().toCharArray();
ArrayList<Pair>res[]=new ArrayList[3*n*m];
for(int i=0;i<3*n*m;i++)
res[i]=new ArrayList();
int k=0;
for(int i=0;i<n;i++)
{
    for(int j=0;j<m;j++)
    {
        ArrayList<Pair>arr=new ArrayList<Pair>();
        ArrayList<Pair>arr2=new ArrayList<Pair>();
        if(i+1<n&&j+1<m)
        {
            if(board[i][j]=='1')
                arr.add(new Pair(i,j));
            else arr2.add(new Pair(i,j));
            if(board[i][j+1]=='1')
                arr.add(new Pair(i,j+1));
            else  arr2.add(new Pair(i,j+1));
            
            if(board[i+1][j]=='1')
                arr.add(new Pair(i+1,j));
            else arr2.add(new Pair(i+1,j));
            if(board[i+1][j+1]=='1')
                arr.add(new Pair(i+1,j+1));
            else arr2.add(new Pair(i+1,j+1));
            if(arr.size()==3)
            {
                for(Pair p:arr){
                    board[p.x][p.y]='0';
                    res[k].add(new Pair(p.x,p.y));
                }
                k++;
            }
            else if(arr.size()==2)
            {
                for(Pair p:arr2)
                {
                    board[p.x][p.y]='1';
                     res[k].add(new Pair(p.x,p.y));
                }
                //k++;
                Pair p=arr.get(0);
                board[p.x][p.y]='0';
                res[k].add(new Pair(p.x,p.y));
                k++;
                for(Pair pp:arr2)
                {
                    board[pp.x][pp.y]='0';
                    res[k].add(new Pair(pp.x,pp.y));
                }
                p=arr.get(1);
                board[p.x][p.y]='0';
                res[k].add(new Pair(p.x,p.y));
                k++;
            }
            else if(arr.size()==1)
            {
                Pair p=arr.get(0);
                Pair tmp=new Pair(p.x,p.y);
                board[p.x][p.y]='0';
                res[k].add(new Pair(p.x,p.y));
                p=arr2.get(1);
                 board[p.x][p.y]='1';
                res[k].add(new Pair(p.x,p.y));
                  p=arr2.get(2);
                 board[p.x][p.y]='1';
                res[k].add(new Pair(p.x,p.y));
                k++;
                arr=new ArrayList();
                arr.add(new Pair(arr2.get(1).x,arr2.get(1).y));
                arr.add(new Pair(arr2.get(2).x,arr2.get(2).y));
                arr2.remove(1);
                arr2.remove(1);
                arr2.add(new Pair(tmp.x,tmp.y));

                for(Pair pp:arr2)
                {
                    board[pp.x][pp.y]='1';
                     res[k].add(new Pair(pp.x,pp.y));
                }
                //k++;
                 p=arr.get(0);
                board[p.x][p.y]='0';
                res[k].add(new Pair(p.x,p.y));
                k++;
                for(Pair pp:arr2)
                {
                    board[pp.x][pp.y]='0';
                    res[k].add(new Pair(pp.x,pp.y));
                }
                p=arr.get(1);
                board[p.x][p.y]='0';
                res[k].add(new Pair(p.x,p.y));
                k++;
            }
            else if(arr.size()==4)
            {
                Pair p=arr.get(0);
                board[p.x][p.y]='0';
                arr2.add(new Pair(p.x,p.y));
                res[k].add(new Pair(p.x,p.y));
                 p=arr.get(1);
                board[p.x][p.y]='0';
                arr2.add(new Pair(p.x,p.y));
                  res[k].add(new Pair(p.x,p.y));

                 p=arr.get(2);
                board[p.x][p.y]='0';
                arr2.add(new Pair(p.x,p.y));
                  res[k].add(new Pair(p.x,p.y));
                  k++;
                arr.remove(0);
                arr.remove(0);
                arr.remove(0);
                p=arr.get(0);

                Pair tmp=new Pair(p.x,p.y);
                board[p.x][p.y]='0';
                res[k].add(new Pair(p.x,p.y));
                p=arr2.get(1);
                 board[p.x][p.y]='1';
                res[k].add(new Pair(p.x,p.y));
                  p=arr2.get(2);
                 board[p.x][p.y]='1';
                res[k].add(new Pair(p.x,p.y));
                k++;
                arr=new ArrayList();
                arr.add(new Pair(arr2.get(1).x,arr2.get(1).y));
                arr.add(new Pair(arr2.get(2).x,arr2.get(2).y));
                arr2.remove(1);
                arr2.remove(1);
                arr2.add(new Pair(tmp.x,tmp.y));

                for(Pair pp:arr2)
                {
                    board[pp.x][pp.y]='1';
                     res[k].add(new Pair(pp.x,pp.y));
                }
                //k++;
                 p=arr.get(0);
                board[p.x][p.y]='0';
                res[k].add(new Pair(p.x,p.y));
                k++;
                for(Pair pp:arr2)
                {
                    board[pp.x][pp.y]='0';
                    res[k].add(new Pair(pp.x,pp.y));
                }
                p=arr.get(1);
                board[p.x][p.y]='0';
                res[k].add(new Pair(p.x,p.y));
                k++;
            }


        }
    }
}
/*for(int i=0;i<n;i++){
for(int j=0;j<m;j++)
{
    out.print(board[i][j]+" ");
}
out.println();
}*/
out.println(k);
for(int i=0;i<k;i++)
{
    for(Pair p:res[i])
        out.print((p.x+1)+" "+(p.y+1)+" ");
    out.println();
}


}




 
out.close();

//SEE UP 

}






 


 
 static class Pair implements Comparable<Pair>{
        public int x, y;
        public Pair(int x1, int y1) {
            x=x1;
            y=y1;
        }
        
        
        


}
 
    
}
