import java.util.*;
import java.io.*;

public class Solution
{
    
    
   
  
  
  
  
  
  static long dp[][];
  
  
 
  
    // Updates a node in Binary Index Tree (BITree)
    // at given index in BITree. The given value 
    // 'val' is added to BITree[i] and all of 
    // its ancestors in tree.
    
    
    
   
    public static void main(String []args) throws IOException
    {
        Scanner sc = new Scanner(System.in);
        int t = sc.nextInt();
       // sc.nextLine();
        while(t-- > 0)
        {
            int n = sc.nextInt();
            sc.nextLine();
            //String s = sc.nextLine();
            int arr[][] = new int[26][n];
            for(int i = 0 ; i < n ; i++)
            {
                String s = sc.nextLine();
                
                for(int j = 0 ; j < 26 ; j++)
                {
                    int cnt = 0;
                    for(int k = 0 ; k < s.length() ; k++)
                    {
                        if(s.charAt(k)-'a' == j)
                        cnt++;
                    }
                    arr[j][i] = 2*cnt-s.length();
                }
            }
            
            int ans = 0;
            for(int i = 0 ; i < 26 ; i++)
            {
             Arrays.sort(arr[i]);   
             int tot = 0;
             for(int j = n-1 ; j >= 0 ;j--)
             {
                 tot += arr[i][j];
                 if(tot <= 0)
                 break;
                 
                 ans = Math.max(ans,n-j);
             }
            }
            System.out.println(ans);
        }
    }
}
