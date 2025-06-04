import java.util.*;
import java.lang.*;
import java.io.*;
import java.math.BigInteger;
import java.text.DecimalFormat;

public final class Solution {

    static int inf = Integer.MAX_VALUE;
    static long mod = 1000000000 + 7;
    
 static void ne(Scanner sc, BufferedWriter op) throws Exception {
            int n=sc.nextInt();
     String one=sc.next();
     String two=sc.next();
     int _10=0;
     int _00=0;
     int _11=0;
     int _01=0;
   for(int i=0;i<n;i++){
       int c1=one.charAt(i)-'0';
       int c2=two.charAt(i)-'0';
       if(c1==0 && c2==0){
           _00++;
       }
       if(c1==1 && c2==0){
           _10++;
       }
       if(c1==1 && c2==1){
           _11++;
       }
       if(c1==0 && c2==1){
           _01++;
       }
   }
   int same=_00+_11;
   int diff=_01+_10;
   if(same==n){
      op.write("0\n");
      return;
   }
   if(same==1 && _11==1){
       op.write("1\n");
       return;
   }
//    if(diff==n){
//        if(diff%2==0){
//            op.write(n+"\n");
 
//        }else{
//            op.write("-1\n");
//        }
//        return;
//    }
   int ans=inf;
    
    if(_11-_00==1 && _00!=0 && _11!=0){
        // print("here");
        ans=Math.min(ans,same);
     }
     if(_10 ==_01 && _10!=0 && _01!=0){
        ans=Math.min(ans,diff);
     }
    
    if(ans==inf){
        op.write("-1\n");
    }else{
        op.write(ans+"\n");
    }

 }
 
  
  
  

    public static void main(String[] args) throws Exception {
        BufferedWriter op = new BufferedWriter(new OutputStreamWriter(System.out));
        // Reader sc = new Reader();
        Scanner sc= new Scanner(System.in);
        int t = sc.nextInt();
        while (t-->0){ ne(sc, op); }
        
        // ne(sc,op);
           
                   
        op.flush();
    }

    
    

    
       

}



// return -1 to put no ahed in array










 
