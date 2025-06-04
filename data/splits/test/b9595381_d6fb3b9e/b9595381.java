import java.io.IOException;
import java.io.InputStreamReader;
import java.util.*;
public class Main{
    
static int dest1;
static int dest2;
public static void main(String args[]){
       	FastScanner in = new FastScanner();
		int test=in.nextInt();
                while(test-->0){
                    int n=in.nextInt();
                    int count[][]=new int[n][5];
                    int total[]=new int[n];
                    String words[]=new String[n];
                    for(int i=0;i<n;i++){
                        words[i]=in.next();
                        for(int j=0;j<words[i].length();j++)
                            count[i][words[i].charAt(j)-'a']++;
                        
                        total[i]=words[i].length();
                    }
                    int max=Integer.MIN_VALUE;
                   for(int i=0;i<5;i++){
                       
                      Integer ans[]=new Integer[n];
                      for(int j=0;j<n;j++){
                          ans[j]=count[j][i]-(total[j]-count[j][i]);
                      }
                      Arrays.sort(ans,Collections.reverseOrder());
                      int j=0;
                      int r=0;
                      while(j<n && r+ans[j]>0){
                          r+=ans[j];
                          j++;
                      }
                      max=Math.max(j,max);
                      
                   }
                   System.out.println(max);
                   
                    
                }
}

       



}


