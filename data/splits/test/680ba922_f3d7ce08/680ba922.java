import java.util.*;
import java.io.*;

public class Solution
{
    
    
   
  
  
  
  
  
  
  
 
  
    // Updates a node in Binary Index Tree (BITree)
    // at given index in BITree. The given value 
    // 'val' is added to BITree[i] and all of 
    // its ancestors in tree.
    
    
    
   

static int dp[][];



    public static void main(String []args) throws IOException
    {
        Scanner sc = new Scanner(System.in);
        int t = sc.nextInt();
        while(t-- > 0)
        {
            int n = sc.nextInt();
            sc.nextLine();
            String a = sc.nextLine();
            String b = sc.nextLine();
            int same = 0 , zo = 0 , oz = 0 , oo = 0 , zz = 0;
            for(int i = 0 ; i < n ; i++)
            {
                if(a.charAt(i) == '0' && b.charAt(i) == '1')
                oz++;
                
                else if(a.charAt(i) == '1' && b.charAt(i) == '0')
                zo++;
                
                else if(a.charAt(i) == '1' && b.charAt(i) == '1')
                oo++;
                
                else
                zz++;
            }
            
            if(oz == zo || (zz == oo-1))
            {
                int mx = Integer.MAX_VALUE;
                if(oz == zo)
                mx = Math.min(mx,2*oz);
                
                if(oo-1 == zz)
                mx = Math.min(mx,zz+oo);
                
                System.out.println(mx);
            }
            
            else
            {
                System.out.println(-1);
            }
        }
    }
}
