import java.io.*;
import java.util.*;
import java.math.BigInteger; 
import java.lang.*;




public class Main {
    
    
    
    
   public static String[] F(BufferedReader bf) throws Exception
    {
        return (bf.readLine().split(" "));
    }
    
   public static void pr(PrintWriter out,Object o)
   {
      
    out.println(o.toString());//out.flush();
    
   }
   
   
   
   
    
    public static void pr(Object o)
    {
        System.out.println(o.toString());
    }
    
    
   
    public static int inInt(String s)
    {
        return Integer.parseInt(s);
    }
    
 
 
    
    
    
   
     
     
        
    
   
    

   
   

   
    
    
   
    public static int F(String[] arr,char ch){


      int[] nums = new int[arr.length];

      for(int i=0;i<arr.length;i++){
        String str = arr[i];int a=0;
        for(int j=0;j<str.length();j++){
              if(str.charAt(j) == ch){
                a++;
              }
        }
        //int b = (str.length()/2);
       // if((str.length()%2) != 0) b++;
        nums[i] = (2*a - (str.length()));
      }

      Arrays.sort(nums);
      int sum = 0;
      
   
      for(int i=arr.length-1;i>=0;i--){
        if((sum+nums[i]) <= 0){
          return  arr.length-1-i;
        }
        sum += nums[i];
      }

      return arr.length;
    }
    public static void main (String[] args) throws Exception {
        
        BufferedReader bf=new BufferedReader(new InputStreamReader(System.in));
        PrintWriter out = new PrintWriter(System.out);;;//
        
        //int[] map=new int[1000001];
      
       int yy=inInt(bf.readLine());
       


        

        for(int w=0;w<yy;w++)
        {
             //String str = bf.readLine();
          out.flush();
         
            String[] xlp = bf.readLine().split(" ");
            
            //String st = bf.readLine();
      int n;//boolean bol=false;
           //  m;//long a,b,c;
           long k;long l;
          // int l;
           //int double avg  k;//pr(out,"vbc");
       // boolean bol = false;
           //long mod=1000000000+7
      n =inInt(xlp[0]);//int m = inInt(xlp[1]);//long b=in(xlp[3]);//long k=in(xlp[4]);
      String[] arr = new String[n];

      for(int i=0;i<n;i++){
        arr[i] = bf.readLine();
      }
      int ans=0;
      for(int i=0;i<5;i++){
        int v = F(arr,(char)(i+'a'));
       // pr(out,v+" "+i);
        ans = Math.max(ans,F(arr,(char)(i+'a')));

      }

      pr(out,ans);
      


      

   }
    out.close();//
   bf.close();//
       
}}
