import java.util.*;
import java.lang.*;
import java.io.*;

/* Name of the class has to be "Main" only if the class is public. */
public class Codechef
{
	public static void main (String[] args) throws java.lang.Exception
	{
		Scanner sc = new Scanner(System.in);
		int t = sc.nextInt();
		while(t-- > 0)
		{
		    int n = sc.nextInt();
		    sc.nextLine();
		    int a[][] = new int[26][26];
		    int b[][][] = new int[26][26][26];
		    
		    boolean bl = false;
		    String arr[] = new String[n];
		    for(int i = 0 ; i < n ; i++)
		    {
		        arr[i] = sc.nextLine();
		        
		        if(arr[i].length() == 1)
		        {
		            bl = true;
		        }
		        
		        else if(arr[i].length() == 2)
		        {
		            //a[arr[i].charAt(0)-'a'][arr[i].charAt(1)-'a'] = 1;
		            if(arr[i].charAt(0)==arr[i].charAt(1))
		            bl = true;
		        }
		        
		        else
		        {
		           //b[arr[i].charAt(0)-'a'][arr[i].charAt(1)-'a'][arr[i].charAt(2)-'a'] = 1; 
		           if(arr[i].charAt(0) == arr[i].charAt(2))
		           bl = true;
		        }
		    }
		    
		    if(bl)
		    System.out.println("YES");
		    
		    else
		    {
		        for(int i = 0; i < n ; i++)
		        {
		            if(arr[i].length() == 2)
		            {
		                int p1 = arr[i].charAt(0)-'a';
		                int p2 = arr[i].charAt(1)-'a';
		                
		                if(a[p2][p1] == 1)
		                bl = true;
		                
		                for(int j = 0;  j < 26 ; j++)
		                {
		                    if(b[p2][p1][j] == 1)
		                    bl = true;
		                }
		                a[p1][p2] = 1;
		            }
		            
		            else
		            {
		                 int p1 = arr[i].charAt(0)-'a';
		                int p2 = arr[i].charAt(1)-'a';
		                int p3 = arr[i].charAt(2)-'a';
		                
		                if(a[p3][p2] == 1)
		                bl = true;
		                
		                if(b[p3][p2][p1] == 1)
		                bl = true;
		                
		                b[p1][p2][p3] = 1;
		            }
		        }
		        if(bl)
		        System.out.println("YES");
		        
		        else
		        System.out.println("NO");
		    }
		}
	}
}
