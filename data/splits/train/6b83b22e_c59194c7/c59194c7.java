import java.io.BufferedReader; 
import java.io.IOException; 
import java.io.InputStreamReader; 
import java.util.Scanner; 
import java.util.StringTokenizer;
import java.util.*;
public class codeforcesB{
  
 public static void main(String args[]){
    FastReader sc=new FastReader();
    StringBuilder sb=new StringBuilder();
    int t=sc.nextInt();
    while(t-->0){
      int n=sc.nextInt();
      int ar[]=new int[n];
      int sum=0;
      for(int i=0;i<n;i++){ar[i]=sc.nextInt();}
      if(n%2==0){
          for(int i=0;i<n;i++){
              if(i%2==0){sb.append(-1*ar[i+1]+" ");}
              else{sb.append(ar[i-1]+" ");}
            }
           
        }
      else{
          if(ar[1]+ar[0]!=0){
          sb.append(ar[2]+" "+ar[2]+" "+-1*(ar[1]+ar[0])+" ");}
          else{
              if(ar[2]+ar[1]!=0){
                  sb.append(-1*(ar[2]+ar[1])+" "+ar[0]+" "+ar[0]+" ");
              }
              else{
                  sb.append(ar[1]+" "+-1*(ar[2]+ar[0])+" "+ar[1]+" ");
              }
          }
          for(int i=3;i<n;i++){
              if(i%2==1){sb.append(-1*ar[i+1]+" ");}
              else{sb.append(ar[i-1]+" ");}
            }
        }
      sb.append("\n");
    }
    System.out.print(sb.toString());
   }
}
